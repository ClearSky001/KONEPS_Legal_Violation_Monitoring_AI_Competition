#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""나라장터 자체입찰 공고 법령 위반사항 모니터링 — 제출 스크립트.

설계 요약
---------
LLM에게 24개 항목을 직접 판정시키지 않는다. LLM은 **공고문에서 사실(fact)만 추출**하고,
법령 임계값·상호배타·부재탐지 처리는 전부 파이썬 규칙 엔진이 결정한다.

    1) 결정적 전처리 : 메타 파싱 → 금액 구간 · 계약법 · 낙찰방법 · 경쟁제품 후보 · 정규식 힌트
    2) 섹션 추출     : 참가자격 / 공고개요 / 공동수급 / 설명회 / 제출서류 / 규격(모델명) 발췌
    3) LLM 1회 호출  : 구조화 출력(JSON Schema)으로 사실 25종 + 근거 인용문 추출
    4) 규칙 엔진     : 사실 → v1..v24 (법정 금액 임계값·중복제한 금지·부재탐지 규칙 적용)
    5) 근거 정합화   : 인용문을 원문 부분문자열로 복원 → 500자 절단 → 부재탐지 항목 공란

경로는 PPS_DATA_DIR · PPS_OUTPUT_DIR · PPS_MODEL_DIR 환경변수를 우선한다.
실행 코드는 전부 ``if __name__ == "__main__":`` 아래에서만 시작한다(vLLM spawn 안전).

로컬 확인
    python script.py --mock --input dev.jsonl --data-dir ./data
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import os
import re
import sys
import time
import traceback
import unicodedata
from collections import Counter
from typing import Any, Dict, Iterator, List, Optional, Tuple

# ===================================================================================
# 1. 경로 · 상수
# ===================================================================================
DATA_DIR = os.environ.get("PPS_DATA_DIR", "./data")
OUTPUT_DIR = os.environ.get("PPS_OUTPUT_DIR", "./output")
MODEL_DIR = os.environ.get("PPS_MODEL_DIR", "/opt/models/gemma-4-26B-A4B-it")
# model/ 정적 자산 디렉터리 — 대회 규정상 제출 ZIP 의 model/ 에는 "실행에 필요한 정적 자산만"
# 포함할 수 있다(LLM 가중치·LoRA 제외). 여기서는 Colab logprob 실측으로 튜닝한 판정 임계
# 정책(policy_margin.json) 같은 정적 파일을 담는다. script.py 옆의 model/ 을 기본으로 하고
# PPS_ASSET_DIR 로 재정의할 수 있다. 파일이 없으면 내장 기본값으로 동작한다(회귀 0).
try:
    ASSET_DIR = os.environ.get("PPS_ASSET_DIR") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "model")
except NameError:                                  # __file__ 이 없는 대화형 환경 대비
    ASSET_DIR = os.environ.get("PPS_ASSET_DIR") or "./model"

ITEMS = [f"v{i}" for i in range(1, 25)]
EVID = [f"e{i}" for i in range(1, 25)]
COLUMNS = ["id"] + ITEMS + EVID
ABSENCE = {"v10", "v11", "v16", "v18", "v20"}      # 부재탐지 → 근거문구 항상 공란

DOC_ORDER = ["공고문", "규격서", "과업지시서", "제안요청서", "예외공표서", "기타"]
META_FIELDS = [
    "적용계약법", "업무구분", "계약방법", "낙찰방법", "낙찰하한율",
    "배정예산금액", "입찰추정가격", "소관구분", "공동도급구성방식", "정보화사업여부",
    "세부품명번호목록", "제한지역코드목록", "지역제한여부", "면허업종제한목록", "업종제한여부",
    "조항호내용", "공고게시일자", "개찰예정일자", "긴급공고여부", "입찰방법", "조달방식",
]

SEED = 20260826
MAX_MODEL_LEN = 16384           # 프롬프트(≤14.5k) + 출력(1.5k)  [변경: 12288→16384, 법령 RAG 컨텍스트 여유]
MAX_TOKENS = 1536               # 사실 추출 JSON 출력 예산  [변경: 1024→1536, 근거 문장 절단 완화]
EVIDENCE_MAX = 500
QUANT = "int8_per_channel_weight_only"

# ── 2차 판정(캐스케이드) 샘플링 : 자기일관성 다수결 + logprob 기반 기권 ──────────────
# [추가 2026-09-10]
#  F) 다수결: 온도>0 으로 JUDGE_SAMPLES 회 샘플링해 v(0/1) 다수결을 취한다. 만장일치일 때만
#     원래 신뢰도를 유지하고, 2/3 처럼 갈리면 conf 를 한 단계 낮춰(=high 박탈) 규칙 결과를 남긴다.
#     정책이 모두 conf>=high 를 요구하므로 "2/3 동의 = 개입하지 않음"이 자동으로 성립한다.
#  G) 기권(conformal abstain): v 값 토큰 위치의 logprob 을 읽어 |logP(1)-logP(0)| 이
#     JUDGE_MARGIN_MIN 미만이면 판정을 low 로 강등해 규칙으로 되돌린다(=abstain).
#     JSON 스키마가 v 를 enum[0,1] 로 제약하므로 그 위치는 사실상 이진 결정이다.
#  [2026-09-16b] 기본값 3→1, logprob 10→0. 새 JUDGE_POLICY(v8/v15/v19/v20 mid) 는 그리디 단일 샘플·기권 없음
#     조건에서 실측된 것이다(eval_api.py 는 logprob 을 받을 수 없음). 다수결·기권은 이 정책과 조합해 측정된 적이
#     없고(2026-09-10 제출 LB 0.6537 에 포함됐으나 효과 분리 불가), mid 기준에서는 "불일치 → 한 단계 강등"이
#     오히려 mid 판정을 통과시키는 방향으로 작동할 수 있어 측정 조건과 동일하게 끈다. 판정 런타임도 1/3.
#     env PPS_JUDGE_SAMPLES / PPS_JUDGE_LOGPROBS 로 재활성 가능.
JUDGE_SAMPLES = int(os.environ.get("PPS_JUDGE_SAMPLES", "1"))     # 1 이면 기존 그리디 단일 샘플
JUDGE_TEMP = float(os.environ.get("PPS_JUDGE_TEMP", "0.6"))       # 다수결용 온도(샘플>1 일 때만 사용)
JUDGE_TOP_P = 0.95
JUDGE_LOGPROBS = int(os.environ.get("PPS_JUDGE_LOGPROBS", "0"))   # 0 이면 logprob 기권 비활성
JUDGE_MARGIN_MIN = float(os.environ.get("PPS_JUDGE_MARGIN", "1.5"))  # ln-odds 1.5 ≈ 82:18

# 법정 금액 임계값(원)
GOSI_NATIONAL = 230_000_000     # 국가·물품/용역 고시금액 2억 3천만원
REGION_LIMIT_LOCAL = 500_000_000  # 지방 지역제한 허용 상한(지방규칙 제24조)
BAND_1E = 100_000_000           # 판로지원법 시행령 제2조의2 1억원 경계

# 공동수급 구성원 최소지분율 하한(%) : 지방 5%, 국가(공동이행) 10%
MIN_SHARE_LOCAL = 5.0
MIN_SHARE_NATIONAL = 10.0

# 협상에 의한 계약 — 지방 제안요청 설명 실시 시기(제안서 제출마감일 전일 기산, 일)
NEGO_BRIEF_DAYS = ((1_000_000_000, 40), (100_000_000, 20), (0, 10))

_T0 = time.time()


def log(msg: str) -> None:
    print(f"[pps] {time.time() - _T0:7.1f}s | {msg}", file=sys.stderr, flush=True)


# ===================================================================================
# 2. 데이터 로더 (대회 베이스라인과 동일한 계약)
# ===================================================================================
def _open(path: str):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return io.open(path, "r", encoding="utf-8")


def validate_record(rec: Any) -> None:
    if not isinstance(rec, dict):
        raise ValueError(f"레코드가 object가 아니다: {type(rec).__name__}")
    for k in ("id", "docs", "meta"):
        if k not in rec:
            raise ValueError(f"필수 키 없음: {k}")
    if not isinstance(rec["id"], str) or not rec["id"]:
        raise ValueError("id가 비어 있다")
    if not isinstance(rec["docs"], list) or not rec["docs"]:
        raise ValueError(f"docs가 비어 있다 (id={rec['id']})")
    for d in rec["docs"]:
        if not isinstance(d, dict) or not all(k in d for k in ("doc_id", "type", "text")):
            raise ValueError(f"docs 원소 형식 오류 (id={rec['id']})")
        if not isinstance(d["text"], str):
            raise ValueError(f"docs.text가 문자열이 아니다 (id={rec['id']})")
    if not isinstance(rec["meta"], dict):
        raise ValueError(f"meta가 object가 아니다 (id={rec['id']})")


def normalize(rec: Dict[str, Any]) -> Dict[str, Any]:
    for d in rec.get("docs", []):
        d["text"] = unicodedata.normalize("NFC", d["text"])
        if isinstance(d.get("type"), str):
            d["type"] = unicodedata.normalize("NFC", d["type"])
    return rec


def iter_records(path: str, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    n = 0
    with _open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno} JSON 파싱 실패: {e}") from e
            validate_record(rec)
            yield normalize(rec)
            n += 1
            if limit and n >= limit:
                return


def full_text(rec: Dict[str, Any]) -> str:
    """근거문구 대조용 원문(NFC · 문서 결합)."""
    return "\n".join(d["text"] for d in rec["docs"])


_FLAT_CACHE: Dict[str, str] = {}


def flat_text(rec: Dict[str, Any]) -> str:
    """문서 전문에서 줄바꿈·연속 공백을 하나의 공백으로 접은 텍스트(윈도 검색용)."""
    key = str(rec.get("id") or "")
    if key and key in _FLAT_CACHE:
        return _FLAT_CACHE[key]
    v = re.sub(r"\s+", " ", full_text(rec))
    if key:
        if len(_FLAT_CACHE) > 8:
            _FLAT_CACHE.clear()
        _FLAT_CACHE[key] = v
    return v


def doc_text(rec: Dict[str, Any], types: Tuple[str, ...]) -> str:
    parts = [d["text"] for d in rec["docs"] if d.get("type") in types]
    return "\n".join(parts)


def notice_text(rec: Dict[str, Any]) -> str:
    t = doc_text(rec, ("공고문",))
    return t if t.strip() else full_text(rec)


def spec_text(rec: Dict[str, Any]) -> str:
    return doc_text(rec, ("규격서", "과업지시서", "제안요청서"))


# ===================================================================================
# 3. 메타 파싱 유틸
# ===================================================================================
def to_int(v: Any) -> int:
    if v is None:
        return 0
    if isinstance(v, (int, float)):
        return int(v)
    s = re.sub(r"[^0-9]", "", str(v))
    return int(s) if s else 0


def meta_str(rec: Dict[str, Any], key: str) -> str:
    v = rec.get("meta", {}).get(key)
    return "" if v is None else unicodedata.normalize("NFC", str(v)).strip()


def is_local_law(rec: Dict[str, Any]) -> bool:
    """지방계약법 적용 여부."""
    return "지방" in meta_str(rec, "적용계약법")


def is_small_quote(rec: Dict[str, Any]) -> bool:
    """소액수의(2인 이상 견적) 여부 — 지방의 경우 v2·v6·v7·v8 예외."""
    s = meta_str(rec, "낙찰방법") + " " + meta_str(rec, "계약방법")
    return ("소액수의" in s) or ("견적" in s and "수의" in s)


# [추가] 나라장터 메타 '계약방법'이 경쟁입찰로 입력돼 있어도 공고문 머리말이 수의계약(견적) 공고인 경우가 있다
#        (dev 144: 메타 '제한경쟁' vs 공고문 "물품구매 수의계약 안내공고 … '입찰'은 '견적'으로 일괄변경").
#        판로지원법 시행령 제2조의2 의 규모 제한 의무·제2조의3 의 예외는 모두 '제한경쟁입찰' 전제이므로
#        수의계약이면 규모 제한의 부재(v16·v18)도 과잉제한(v14·v15·v17)도 위반이 아니다.
#        머리말 1,200자로 스코프를 좁혀 본문 상투구("낙찰자 미체결 시 수의계약 가능")를 배제한다.
_RE_SOLE_HEAD = re.compile(
    r"수의계약\s*(?:안내|견적)?\s*공고|소액\s*수의|견적\s*제출\s*안내\s*공고|2인\s*이상[^\n]{0,8}견적")


def is_sole_notice(rec: Dict[str, Any]) -> bool:
    """공고문 머리말이 수의계약(견적) 공고임을 직접 밝히는가."""
    return bool(_RE_SOLE_HEAD.search(notice_text(rec)[:1200]))


def is_negotiation(rec: Dict[str, Any]) -> bool:
    """협상에 의한 계약 여부 (v22·v23 적용 조건)."""
    s = meta_str(rec, "낙찰방법") + " " + meta_str(rec, "계약방법") + " " + meta_str(rec, "조항호내용")
    return "협상" in s


def price_band(est: int) -> str:
    if est < BAND_1E:
        return "under1e"
    if est < GOSI_NATIONAL:
        return "mid"
    return "over_gosi"


def region_threshold(local: bool) -> int:
    """지역제한이 허용되는 추정가격 상한."""
    return REGION_LIMIT_LOCAL if local else GOSI_NATIONAL


# ===================================================================================
# 4. 중기간 경쟁제품 사전
# ===================================================================================
_COMP_CACHE: Dict[str, Any] = {}


def resolve_data_dir(data_dir: str) -> str:
    """`법령패키지`(경쟁제품 CSV·법령 원문)를 실제로 담고 있는 디렉터리를 찾아준다.

    평가 서버는 `법령패키지`·`항목표.json`·`test.jsonl` 이 함께 있는 폴더를 넘기지만,
    로컬에서는 그 상위(`competition/`)를 넘기기 쉽다. 그러면 경쟁제품 사전 615코드와
    법령 색인이 통째로 비어 v12·v14 가 대량 오발화한다(dev200 0.8697 → 0.8511).
    조용히 성능이 깎이는 실수이므로, 상위/하위 한 단계까지 훑어 보정하고 로그를 남긴다.
    """
    if not data_dir:
        return data_dir
    marker = "법령패키지"
    if os.path.isdir(os.path.join(data_dir, marker)):
        return data_dir
    for cand in (os.path.join(data_dir, "data"), os.path.dirname(os.path.abspath(data_dir))):
        if cand and os.path.isdir(os.path.join(cand, marker)):
            log(f"data-dir 보정: {data_dir} → {cand} ({marker} 발견)")
            return cand
    log(f"주의: {data_dir} 아래에서 {marker} 를 찾지 못했습니다 — 경쟁제품·법령 사전 없이 진행합니다")
    return data_dir


def load_competition_products(data_dir: str) -> Tuple[Dict[str, str], List[Tuple[str, str]]]:
    """중기부고시 경쟁제품 세부품명 CSV → (코드→품명, [(품명, 특이사항)])."""
    if _COMP_CACHE:
        return _COMP_CACHE["codes"], _COMP_CACHE["names"]
    codes: Dict[str, str] = {}
    notes: Dict[str, str] = {}
    names: List[Tuple[str, str]] = []
    base = os.path.join(data_dir, "법령패키지", "중기부고시")
    cand: List[str] = []
    if os.path.isdir(base):
        cand = [os.path.join(base, f) for f in sorted(os.listdir(base)) if f.endswith(".csv")]
    for path in cand:
        try:
            with io.open(path, encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    code = (row.get("세부품명번호") or "").strip()
                    name = unicodedata.normalize("NFC", (row.get("세부품명") or "").strip())
                    note = (row.get("특이사항") or "").strip()
                    if code:
                        codes[code] = name
                        notes[code] = note
                    if name:
                        names.append((name, note))
        except Exception as e:                                     # 파일이 없거나 형식이 달라도 계속 진행
            log(f"경쟁제품 CSV 로드 실패({os.path.basename(path)}): {e}")
    _COMP_CACHE["codes"], _COMP_CACHE["names"], _COMP_CACHE["notes"] = codes, names, notes
    log(f"경쟁제품 세부품명 {len(codes)}코드 / {len(names)}품명 로드")
    return codes, names


_NOTE_LT = re.compile(r"추정\s*가격\s*([0-9][0-9,.]*)\s*(억원|억|천만원|백만원|만원|원)?\s*미만")
_NOTE_GE = re.compile(r"추정\s*가격\s*([0-9][0-9,.]*)\s*(억원|억|천만원|백만원|만원|원)?\s*이상")
_NOTE_UNIT = {"억원": 10 ** 8, "억": 10 ** 8, "천만원": 10 ** 7, "백만원": 10 ** 6,
              "만원": 10 ** 4, "원": 1, None: 10 ** 8, "": 10 ** 8}


def note_amount_ok(note: str, est: int) -> bool:
    """중기부고시 '특이사항'의 추정가격 조건('3억원 미만에 한함' 등)을 만족하는지."""
    if not note or est <= 0:
        return True
    m = _NOTE_LT.search(note)
    if m:
        try:
            lim = float(m.group(1).replace(",", "")) * _NOTE_UNIT.get(m.group(2), 10 ** 8)
        except ValueError:
            return True
        return est < lim
    m = _NOTE_GE.search(note)
    if m:
        try:
            lim = float(m.group(1).replace(",", "")) * _NOTE_UNIT.get(m.group(2), 10 ** 8)
        except ValueError:
            return True
        return est >= lim
    return True


_RE_DPC_LINE = re.compile(r"[^\n]{0,160}직접\s*생산\s*확인[^\n]{0,160}")


def competition_name_in_dpc(full: str, names: List[Tuple[str, str]]) -> bool:
    """직접생산확인 문장 안에 경쟁제품 세부품명(공백 무시, 6자 이상)이 그대로 적혀 있는가."""
    if not full or not names:
        return False
    lines = [re.sub(r"\s+", "", m.group(0)) for m in _RE_DPC_LINE.finditer(full)]
    # [변경] 물품 인증서를 대체서류 목록으로만 언급하는 문장(예: "직접생산확인증 또는 환경마크 또는 GR")은 제외.
    #        무라벨 표본에서 폐아스콘 처리 용역이 물품 경쟁제품명과 매칭돼 v11 오발화 → '증명서' 또는 '판로지원' 인용 문장만 인정.
    lines = [ln for ln in lines if ("직접생산확인증명서" in ln or "직접생산증명서" in ln or "판로지원" in ln)]
    if not lines:
        return False
    for name, _note in names:
        n = re.sub(r"\s+", "", name)
        if len(n) >= 6 and any(n in ln for ln in lines):
            return True
    return False


def competition_notes() -> Dict[str, str]:
    return _COMP_CACHE.get("notes", {})


def _bigrams(s: str) -> set:
    s = re.sub(r"[^0-9A-Za-z가-힣]", "", s)
    return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) > 1 else ({s} if s else set())


def competition_candidates(title: str, names: List[Tuple[str, str]], topk: int = 6) -> List[str]:
    """공고 제목과 문자 바이그램이 겹치는 경쟁제품 세부품명 후보."""
    tb = _bigrams(title)
    if not tb or not names:
        return []
    scored = []
    for name, _note in names:
        nb = _bigrams(name)
        if not nb:
            continue
        inter = len(tb & nb)
        if inter >= 2:
            scored.append((inter / len(nb), inter, name))
    scored.sort(reverse=True)
    out, seen = [], set()
    for _s, _i, name in scored:
        if name not in seen:
            seen.add(name)
            out.append(name)
        if len(out) >= topk:
            break
    return out


# ===================================================================================
# 5. 섹션 추출
# ===================================================================================
_QUAL_HEAD = re.compile(
    r"(?:^|\n)[^\n]{0,20}(?:입찰\s*)?참가\s*(?:자격|적격|자\s*자격)[^\n]{0,30}(?:\n|$)")
_SECTION_BREAK = re.compile(
    r"(?:^|\n)\s*(?:[0-9]{1,2}\s*[.)]|[가-힣]\s*[.)]|[IVX]+\s*[.)]|제\s*[0-9]+\s*[장절])\s*"
    r"(?:입찰\s*서|낙찰자|계약\s*체결|입찰\s*보증|청렴|유의\s*사항|기타|과업|제안서\s*평가|현장\s*설명|"
    r"입찰\s*무효|적격\s*심사|평가\s*방법|제출\s*서류|공동\s*계약)")



def competition_best_sim(title: str, names: List[Tuple[str, str]]) -> float:
    """공고 제목과 경쟁제품 세부품명 사이의 최대 바이그램 포함도(0~1).

    세부품명번호가 공고문에 표기되지 않는 서비스성 경쟁제품(행사기획및대행서비스 등)을
    잡아내기 위한 보조 신호. 임계값 0.5는 dev 셋에서 재현율/오탐 균형이 가장 좋았다.
    """
    tb = _bigrams(title)
    if not tb or not names:
        return 0.0
    best = 0.0
    for name, _note in names:
        nb = _bigrams(name)
        if not nb:
            continue
        inter = len(tb & nb)
        if inter >= 2:
            best = max(best, inter / len(nb))
    return best

def clip(s: str, n: int) -> str:
    s = s.strip()
    return s if len(s) <= n else s[:n] + " …(생략)"


# [추가·§10-4 섹션인식] "참가자격"이라는 낱말이 나온다고 모두 자격 조항은 아니다.
#   · 청렴서약서 "입찰참가자격 제한 처분을 받겠습니다"
#   · 부정당업자 안내 "참가 자격이 제한됩니다"
#   · 등록 안내 "참가자격등록증을 변경등록하고"
# 이런 줄이 문서 앞쪽에 있으면 블록 예산(3.6k)을 먼저 먹어치워 정작 뒤에 있는 진짜
# 자격 조항(중소기업·소상공인 제한 등)이 잘려 나간다 → 규모제한을 못 찾고 v16·v18
# 부재탐지가 오발화한다(dev200 v16 FP 132·149 가 정확히 이 경우).
# 주의: 진짜 자격 조항 안에도 "입찰참가자격 제한 대상이 아닐 것"(결격사유) 줄이 있으므로
#       '제재를 받겠다/제한된다'는 서술형 문장만 걸러야 한다.
_RE_QUAL_NOISE = re.compile(
    r"제한\s*처분을?\s*받|자격이?\s*제한\s*(?:됩니다|된다|될\s*수)|참가자격\s*등록증|"
    r"서약(?:서|합니다|하며)|제재\s*처분을?\s*받|무효\s*입찰|변경\s*등록")


def _qual_hit_is_noise(text: str, pos: int) -> bool:
    """해당 '참가자격' 언급이 자격 조항이 아니라 서약·제재·등록 안내문인지."""
    ls = text.rfind("\n", 0, pos) + 1
    le = text.find("\n", pos)
    line = text[ls: le if le > 0 else min(len(text), pos + 200)]
    return bool(_RE_QUAL_NOISE.search(line))


def qualification_section_found(text: str) -> bool:
    """서약·제재 안내가 아닌 '진짜' 입찰참가자격 조항이 문서에 있는지.

    부재탐지(v16·v18: 규모제한이 '없다'는 주장)는 자격 조항을 실제로 읽었을 때만
    성립한다. 자격 섹션 자체가 없는 공고(dev200 v16 FP 149: 4개 히트가 모두
    서약서·부정당업자 안내)에서 '제한이 없다'고 주장하면 근거 없는 오발화가 된다.
    """
    for m in re.finditer(r"참가\s*자격|참가자격|자격\s*요건|참가\s*요건", text):
        if not _qual_hit_is_noise(text, m.start()):
            return True
    return False


# [추가] '중소기업·소상공인 확인서' 제출을 요구하면 규모 제한이 사실상 존재한다.
# 제출서류 목록에만 적혀 자격 조항 정규식에 안 걸리는 공고가 있다(dev200 v16 FP 132).
# 부재탐지(v16·v18)에서만 쓰는 보수적 신호 — 위반을 '주장'하는 데는 쓰지 않는다.
_RE_SIZE_DOC = re.compile(
    r"(?:중\s*[·․ㆍ.]?\s*소기업|소기업|소상공인)[^\n]{0,20}확인서"
    r"|확인서[^\n]{0,20}(?:중\s*[·․ㆍ.]?\s*소기업|소기업|소상공인)")


def extract_qualification(text: str, max_chars: int = 3600) -> str:
    """입찰참가자격 조항 블록. 여러 곳에 흩어져 있으면 모두 모은다."""
    hits = [m.start() for m in re.finditer(r"참가\s*자격|참가자격|자격\s*요건|참가\s*요건", text)]
    clean = hits
    # 전부 노이즈로 걸러지면(자격 조항이 서약문 안에만 있는 공고) 원래 히트로 되돌린다.
    hits = clean or hits
    if not hits:
        return ""
    blocks, used = [], 0
    last_end = -1
    for s in hits:
        if s < last_end:                                          # 이미 담은 구간
            continue
        seg = text[s: s + 2400]
        # [변경] 절 제목만 경계로 삼는다. "2) 과업지시서 상의 과업을 이행할 수 있는 업체" 같은 긴 세부항목
        #        줄은 제목이 아니다 (dev 060: 이 줄에서 잘려 뒤의 직접생산증명서 요건을 놓쳤다).
        for brk in _SECTION_BREAK.finditer(seg, 200):
            nl = seg.find("\n", brk.end())
            if len(seg[brk.start(): nl if nl > 0 else len(seg)].strip()) <= 60:
                seg = seg[: brk.start()]
                break
        seg = seg.strip()
        if len(seg) < 40:
            continue
        blocks.append(seg)
        last_end = s + len(seg)
        used += len(seg)
        if used >= max_chars:
            break
    return clip("\n---\n".join(blocks), max_chars)


def extract_around(text: str, pattern: str, width: int = 420, limit: int = 3, cap: int = 1400) -> str:
    out, used = [], 0
    for m in re.finditer(pattern, text):
        s = max(0, m.start() - width // 3)
        seg = text[s: m.start() + width].strip()
        if seg and all(seg[:60] not in o for o in out):
            out.append(seg)
            used += len(seg)
        if len(out) >= limit or used >= cap:
            break
    return clip("\n…\n".join(out), cap)


def notice_head(text: str, n: int = 1400) -> str:
    """공고 개요(입찰에 부치는 사항) — 금액·기간·계약방법이 들어 있는 앞부분."""
    m = re.search(r"입찰에\s*부치는\s*사항|입찰\s*개요|사업\s*개요|계약\s*개요", text)
    start = m.start() if m else 0
    return clip(text[start: start + n], n)


def guess_title(text: str) -> str:
    m = re.search(r"(?:건\s*명|사\s*업\s*명|용\s*역\s*명|공\s*고\s*명|물\s*품\s*명|구\s*매\s*명|과\s*업\s*명)"
                  r"\s*[:：]?\s*([^\n|]{4,90})", text)
    if m:
        return m.group(1).strip()
    for line in text.split("\n")[:40]:
        line = line.strip()
        if 6 <= len(line) <= 90 and re.search(r"(용역|구매|공사|사업|임차|제작|운영)", line):
            return line
    return text[:60].replace("\n", " ")


# ===================================================================================
# 6. 결정적 힌트(정규식) — LLM 보조 및 실패 시 대체 근거
# ===================================================================================
_SHARE_PAT = [
    re.compile(r"(?:최소\s*)?(?:계약참여\s*)?(?:지분율|출자비율|참여비율|지분|출자\s*비율)"
               r"[^\n]{0,40}?([0-9]{1,2}(?:\.[0-9])?)\s*(?:%|퍼센트|프로)"),
    re.compile(r"([0-9]{1,2}(?:\.[0-9])?)\s*%\s*이상[^\n]{0,20}(?:지분|출자|참여비율)"),
]
_AMOUNT_UNIT = re.compile(
    r"([0-9][0-9,]*(?:\.[0-9]+)?)\s*(억원|억|천만원|백만원|만원|원)")
_DATE_PAT = re.compile(r"(20[0-9]{2})\s*[.\-/년]\s*([0-9]{1,2})\s*[.\-/월]\s*([0-9]{1,2})")


def parse_share_ratios(text: str) -> List[float]:
    vals: List[float] = []
    for pat in _SHARE_PAT:
        for m in pat.finditer(text):
            try:
                v = float(m.group(1))
            except ValueError:
                continue
            if 0 < v <= 100:
                vals.append(v)
    return vals


def parse_amount_won(s: str) -> int:
    """'3억원', '1억 5천만원', '300,000,000원' 등 → 원 단위 정수."""
    total, matched = 0, False
    for m in _AMOUNT_UNIT.finditer(s):
        try:
            num = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        unit = m.group(2)
        mul = {"억원": 1e8, "억": 1e8, "천만원": 1e7, "백만원": 1e6, "만원": 1e4, "원": 1}[unit]
        if unit == "원" and num < 10000:                     # '3원' 같은 오검출 방지
            continue
        total += int(num * mul)
        matched = True
        if unit != "원" and total >= 1e8:
            break
    return total if matched else 0


def parse_dates(s: str) -> List[Tuple[int, int, int]]:
    out = []
    for m in _DATE_PAT.finditer(s):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            out.append((y, mo, d))
    return out


def ordinal_day(ymd: Tuple[int, int, int]) -> int:
    """윤년을 반영한 일련 일수(간이 그레고리력)."""
    y, m, d = ymd
    if m <= 2:
        y -= 1
        m += 12
    return int(365.25 * y) - y // 100 + y // 400 + int(30.6001 * (m + 1)) + d


def days_between(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> int:
    return ordinal_day(b) - ordinal_day(a)


def detect_meta_amount_mismatch(rec: Dict[str, Any]) -> Tuple[bool, str]:
    """공고문이 밝힌 추정가격과 나라장터 메타 추정가격의 불일치."""
    est = to_int(rec["meta"].get("입찰추정가격"))
    if est <= 0:
        return False, ""
    text = notice_text(rec)
    # 본문 어딘가에 메타 추정가격이 그대로 적혀 있으면 메타와 어긋난 것이 아니다.
    # (표 형식 "기초금액 | 추정가격 | 부가가치세 | 91,800,000 | 83,454,545" 에서 첫 숫자만 읽어 생긴
    #  dev FP 059·097·122 · 무라벨 20k 발화 5.8% 의 상당수)
    if re.search(r"(?<![0-9])" + f"{est:,}" + r"(?![0-9])", text) or \
            re.search(r"(?<![0-9,])" + str(est) + r"(?![0-9])", text):
        return False, ""
    vat_tol = max(1000, int(est * 0.005))
    for m in re.finditer(r"(?<![0-9])([0-9][0-9,]{5,})(?![0-9])", text):
        try:
            val = int(m.group(1).replace(",", ""))
        except ValueError:
            continue
        if abs(val - est) <= vat_tol or abs(val - int(round(est * 1.1))) <= vat_tol:
            return False, ""
    cands: List[Tuple[int, str]] = []
    for m in re.finditer(r"추정\s*가\s*격\s*(?:\([^)]{0,6}\))?\s*([^0-9]{0,20})([0-9][0-9,]{5,})", text):
        if re.search(r"[+＋]|포함|기준|이상|미만|초과|이하", m.group(1)):    # "(추정가격+부가가치세)" · "추정가격 기준" 은 금액 표기가 아니다
            continue
        try:
            cands.append((int(m.group(2).replace(",", "")), text[max(0, m.start() - 30): m.end() + 20]))
        except ValueError:
            pass
    if not cands:
        return False, ""
    tol = max(1000, int(est * 0.005))
    if all(abs(c - est) > tol for c, _ in cands):
        return True, cands[0][1].strip()
    return False, ""

# --- 지역제한 결정적 검출기 ----------------------------------------------------
# 나라장터 공고문·첨부문서는 지역제한을 "본점/주된 영업소가 ○○에 소재한 업체"라는
# 정형 문장으로 표현한다. 익명화 마커([지역:r1|단위=기초|광역=경기도])도 함께 인식한다.
SIDO_MAP: Dict[str, str] = {
    "서울특별시": "서울", "서울시": "서울", "서울": "서울",
    "부산광역시": "부산", "부산시": "부산", "부산": "부산",
    "대구광역시": "대구", "대구시": "대구", "대구": "대구",
    "인천광역시": "인천", "인천시": "인천", "인천": "인천",
    "광주광역시": "광주", "광주시": "광주",
    "대전광역시": "대전", "대전시": "대전", "대전": "대전",
    "울산광역시": "울산", "울산시": "울산", "울산": "울산",
    "세종특별자치시": "세종", "세종시": "세종",
    "경기도": "경기", "경기": "경기",
    "강원특별자치도": "강원", "강원도": "강원", "강원": "강원",
    "충청북도": "충북", "충북": "충북", "충청남도": "충남", "충남": "충남",
    "전북특별자치도": "전북", "전라북도": "전북", "전북": "전북",
    "전라남도": "전남", "전남": "전남",
    "경상북도": "경북", "경북": "경북", "경상남도": "경남", "경남": "경남",
    "제주특별자치도": "제주", "제주도": "제주", "제주": "제주",
}
SIDO_PAT = re.compile("|".join(sorted(SIDO_MAP, key=len, reverse=True)))
REG_KEY = re.compile(
    r"본점|본사|주된\s*영업소|주된\s*사무소|주사무소|영업소|사무소\s*소재지|사업장의?\s*소재지|"
    r"관할\s*구역|지역\s*제한|지역제한|소재지를\s*계속|소재한\s*업체|소재하는\s*업체|에\s*소재한|"
    r"지역\s*업체|지역의\s*업체|소재\s*업체")
REG_VERB = re.compile(
    r"둔\s*(?:업체|자|업체여야)|두고|소재하고|소재한|소재하는|기재되어\s*있는|있는\s*업체|"
    r"있어야|인\s*업체|안에\s*있|관내에\s*있|제한")
REG_BASIC = re.compile(r"단위=기초|기초자치단체|시·군·구|시군구|시,\s*군,\s*구")
# [추가] "…에 본점을 둔 업체가 단독 입찰한 경우 배점한도(3점)를 적용하여 평가" 류 적격심사 가점 줄은 참가자격 지역제한이 아니다(무라벨 20k v8 FP).
REG_NOISE = re.compile(
    r"제출\s*장소|등록\s*장소|주\s*소\s*[:：]|우편|전화번호|팩스|계좌|납품\s*장소|설명회\s*장소|개찰\s*장소"
    r"|배점\s*한도|배점을?\s*(?:적용|부여)|가산점|가점을?\s*(?:부여|적용)"
    # [추가] 서약서 본문(입찰건명에 '기초자치단체' 포함)은 참가자격 문장이 아니다(무라벨 20k v6 FP).
    r"|서\s*약\s*서|이의가\s*없음을\s*확약")


# "경상남도내"·"부산광역시에 소재"·"경기도 관내" 처럼 광역 명 바로 뒤에 범위 조사가 붙은 광역 단위 표현
_RE_SIDO_WIDE = re.compile(
    rf"(?:{SIDO_PAT.pattern})\s*(?:내|관내|지역\s*내|에\s*(?:소재|주된|본점|위치)|소재|안에)"
    # "지역제한(경상남도)"·"소재지가 부산광역시)" — 괄호 안에 광역 명만 적은 표기(무라벨 20k v6 FP)
    rf"|지역\s*제한\s*\(\s*(?:{SIDO_PAT.pattern})\s*\)"
    rf"|소재지[가는]\s*(?:{SIDO_PAT.pattern})\s*\)")


# 참가자격 지역제한 문장에만 나오는 강한 표지(본점·주된 영업소·지역제한). 과업 위치·수요기관 소개 문장에는 없다.
REG_STRONG = re.compile(r"본점|본사|주된\s*영업소|주된\s*사무소|주\s*된\s*사무소|주사무소|지역\s*제한|지역제한|영업소의?\s*소재지")
# "[수요기관(기초자치단체)]에서 제시하는" 류 익명화 마커는 수요기관 소개일 뿐 지역제한 단위가 아니다(무라벨 20k v6 FP 5건).
# 단, "본점소재지가 [수요기관(기초자치단체)]내에 소재"(dev 071) 처럼 범위 조사가 바로 붙으면 지역 자체이므로 남긴다.
_RE_DEMAND_MARK = re.compile(
    r"\[수요기관\s*\([^)]*\)[^\]]*\](?!\s*(?:내|관내|안에|에\s*소재|지역))"
    r"|수요기관\s*\([^)]*\)(?![\]|]|\s*(?:내|관내|안에|에\s*소재|지역))")


def detect_region(rec: Dict[str, Any]) -> Dict[str, Any]:
    """지역제한 여부/단위(기초)/복수광역 확대 여부를 문서 전문과 메타에서 결정적으로 판정."""
    reg = basic = multi = False
    basic_strong = wide_strong = False   # 강한 표지가 있는 문장에서의 기초/광역 판정
    reg_strong = False
    quote = ""
    for raw in full_text(rec).split("\n"):
        s = raw.strip()
        if len(s) < 18 or len(s) > 600:
            continue
        if REG_NOISE.search(s):
            continue
        if not REG_KEY.search(s) or not REG_VERB.search(s):
            continue
        s_chk = _RE_DEMAND_MARK.sub(" ", s)
        names = {SIDO_MAP[m.group(0)] for m in SIDO_PAT.finditer(s_chk)}
        b = bool(REG_BASIC.search(s_chk))
        if not names and not b:
            continue
        reg = True
        strong = bool(REG_STRONG.search(s_chk))
        # 지역제한여부=N 인 공고에서의 본문 판정용: 강한 표지 또는 '업체·참가자격' 을 언급하는 자격 문장이어야 한다
        # (dev 048 "경북에 소재한 실적이 우수한 업체" 는 정답 1, 과업 위치·성과품 문장은 '업체' 가 없다).
        reg_strong = reg_strong or strong or bool(re.search(r"업체|참가\s*자격|자격을?\s*갖춘", s_chk))
        if not quote:
            quote = s[:250]
        basic = basic or b
        if strong:
            basic_strong = basic_strong or b
            wide_strong = wide_strong or (not b and bool(_RE_SIDO_WIDE.search(s_chk)))
        multi = multi or (len(names) >= 2)
    # [변경] 자격 문장이 "경상남도내에 주된 영업소"처럼 광역으로 명시돼 있으면, 과업 위치 문장의
    #        '[지역:r1|단위=기초]' 마커로 기초 단위를 판정하지 않는다(무라벨 20k v6 FP).
    if wide_strong and not basic_strong:
        basic = False
    codes = meta_str(rec, "제한지역코드목록")
    # [변경] 메타 지역제한여부=N 이고 제한지역코드도 없으면, 강한 표지(본점·주된 영업소·지역제한) 문장이 있어야
    #        지역제한으로 본다 — 과업 위치·서약서 문장의 지역 마커 오탐 방지(dev 08·071은 모두 강한 표지 보유).
    if reg and not codes and meta_str(rec, "지역제한여부").strip().upper() == "N" and not reg_strong:
        reg = basic = multi = False
        quote = ""
    if codes:
        reg = True
        # [변경] 본문 자격 문장이 "경상남도내"·"부산광역시" 등 광역 단위로 명시돼 있으면 메타의 '단위=기초' 태그
        #        (서약서 수신처·과업 위치 등에서 유입)보다 본문을 우선한다(무라벨 20k v6 FP).
        wide_only = (bool(quote) and not REG_BASIC.search(_RE_DEMAND_MARK.sub(" ", quote))
                     and bool(_RE_SIDO_WIDE.search(quote))) or (wide_strong and not basic_strong)
        if "단위=기초" in codes and not wide_only:
            basic = True
        if len({SIDO_MAP[m.group(0)] for m in SIDO_PAT.finditer(codes)}) >= 2:
            multi = True
    return {"reg": reg, "basic": basic, "multi": multi, "quote": quote}


# --- 소프트웨어사업 결정적 검출기 -----------------------------------------------
# 「소프트웨어 진흥법」상 SW사업은 면허업종제한 '소프트웨어사업자(1468)' 또는
# 본문의 '소프트웨어사업자' 자격 요구로 판별한다. 대기업 참여제한 하한제도를
# 이미 명시한 공고는 v20(미명시) 대상에서 제외한다.
SW_LICENSE = re.compile(r"소프트웨어\s*사업자|1468")
SW_LIMIT = re.compile(
    r"소프트웨어\s*진흥법[^\n]{0,20}제\s*48\s*조|상호출자제한기업집단|대기업[^\n]{0,40}참여\s*제한|"
    r"대기업인?\s*소프트웨어\s*사업자|중견\s*기업[^\n]{0,30}참여"
    # [추가] "대기업 및 중견기업은 입찰에 참여할 수 없음"·"중소 소프트웨어사업자의 사업 참여 지원에 관한 지침이 적용되는 사업"
    #        (무라벨 20k v20 FP 3건) — 참여제한 '적용 여부'가 명시된 것이다.
    r"|대기업[^\n]{0,40}(?:참여|참가)[^\n]{0,12}(?:할\s*수\s*없|불가|금지|제한)"
    r"|중소\s*소프트웨어\s*사업자의\s*사업\s*참여\s*지원")
# 물품 공고에서 소프트웨어사업자 등록만 요구하는 하드웨어 구매(CCTV·GPGPU·메인프레임·스토리지·사이렌 등)는 SW사업이 아니다.
# 단, 라이선스·사용권·솔루션 구매는 dev 133(RHEL 라이선스, 정답 1)처럼 대상이므로 건명으로 구분한다.
_RE_SW_GOODS_TITLE = re.compile(r"소프트웨어|SW|S/W|라이선스|라이센스|사용권|솔루션|프로그램|시스템|플랫폼|패키지|유지관리|유지보수|개발", re.I)


def detect_software(rec: Dict[str, Any]) -> Dict[str, Any]:
    text = full_text(rec)
    lic = meta_str(rec, "면허업종제한목록")
    sw = bool(SW_LICENSE.search(lic)) or ("소프트웨어사업자" in text)
    m = SW_LIMIT.search(text)
    quote = ""
    if sw:
        quote = line_with(text, r"소프트웨어\s*사업자") if "소프트웨어사업자" in text else ""
    return {"sw": sw, "limit": bool(m), "quote": quote}

# --- 물품공급·기술지원 확약서 검출기(v19 보조) -----------------------------------
PLEDGE_CTX = re.compile(
    r"물품\s*공급|공급\s*증명원|공급\s*(?:및|·|,)?\s*(?:기술)?지원|기술\s*지원|A\s*/?\s*S|정품인증|"
    r"제조사|공급사|제조회사|제조업체|제조자")
PLEDGE_STAGE = re.compile(
    r"입찰\s*(?:서)?\s*제출\s*마감일?\s*전|전자입찰서\s*제출\s*마감\s*전|입찰\s*전까지|입찰\s*참가\s*자격|"
    r"입찰관련서류|입찰\s*참가\s*신청|입찰\s*대상자에서\s*제외|미제출")
PLEDGE_NEG = re.compile(
    r"청렴|서약서|담합|부정당|보증금|제출할\s*수\s*있는|낙찰예정자|개찰\s*이후|과업수행자|성과품")


# 위반의 성립요건은 "입찰서 제출 마감 전까지 보유·제출"이라는 시점 강제다.
# 계약 시 제출·제출서류 목록 나열만으로는 위반이 아니므로 시점 문구를 필수로 본다.
# dev 200건: TP4/FP0 (P 1.000 · R 0.667) — 느슨한 검출기(P 0.417)보다 훨씬 낫다.
# [변경] 시점 문구~보유/제출 사이 허용 간격 40→80자: "마감일 전일까지 제조사로부터 당해 물품공급 및
#        무상지원(A/S) 확약서를 발급후 보유"(dev 035, 정답 1) 처럼 목적어가 긴 문장을 놓치고 있었다.
PLEDGE_STRONG = re.compile(
    r"(?:입찰\s*)?(?:전자)?입찰서?\s*제출\s*마감일?\s*전(?:일|까지)?[^\n]{0,80}(?:보유|제출)"
    r"|입찰\s*전(?:까지)?[^\n]{0,30}(?:보유|제출)"
    r"|확약서[^\n]{0,60}보유(?:하여야|해야|하고)")


def _sentence_around(t: str, a: int, b: int, back: int = 200, fwd: int = 120) -> str:
    """위치 a~b 를 포함하는 문장(직전 '다. '/'음. '/줄바꿈 ~ 직후 '. '/줄바꿈) 을 돌려준다."""
    lo = max(0, a - back)
    seg = t[lo:a]
    cut = max(seg.rfind(". "), seg.rfind("\n"), seg.rfind("다."), seg.rfind("함."))
    start = lo + cut + 1 if cut >= 0 else lo
    hi = min(len(t), b + fwd)
    seg2 = t[b:hi]
    m2 = re.search(r"\. |\n|다\.|함\.", seg2)
    end = b + m2.end() if m2 else hi
    return t[start:end]


def detect_pledge(rec: Dict[str, Any]) -> Tuple[bool, str]:
    t = flat_text(rec)
    for m in re.finditer(r"확약서", t):
        w = t[max(0, m.start() - 260): m.end() + 200]
        # [변경] 부정 문맥(청렴·부정당·보증금 …)은 확약서가 든 '문장' 안에서만 본다. 이전엔 260자 창을
        #        써서 앞 조항의 "부정당업자" 한 단어가 정상적인 물품공급 확약서 요구(dev 035)를 지웠다.
        wn = _sentence_around(t, m.start(), m.end())
        if PLEDGE_NEG.search(wn) or not PLEDGE_CTX.search(w) or not PLEDGE_STAGE.search(w):
            continue
        return True, t[max(0, m.start() - 150): m.end() + 60]
    return False, ""


def detect_pledge_strong(rec: Dict[str, Any]) -> Tuple[bool, str]:
    """확약서를 '입찰 단계'에 요구하는 문장만 잡는 정밀 검출기."""
    if not detect_pledge(rec)[0]:
        return False, ""
    t = flat_text(rec)
    for m in re.finditer(r"확약서|공급\s*증명원", t):
        w = t[max(0, m.start() - 300): m.end() + 250]
        if PLEDGE_STRONG.search(w):
            return True, t[max(0, m.start() - 150): m.end() + 60]
    return False, ""


# --- 설명회 참석 강제(v22) / 설명회~제안서마감 기간(v23) 검출기 ------------------
BRIEF_HIT = re.compile(r"(?:사업|과업|현장|제안요청|제안|입찰)?\s*설명회|현장\s*설명")
BRIEF_NEG = re.compile(r"제안서\s*설명|제안\s*설명|발표|프레젠테이션|결과\s*설명회|중간보고|착수보고")
# [변경] 자가라벨 2,112건 대조에서 v22 규칙 FP 18/40 → 원인별 보강:
#   ① "불참에 따른 불이익은 없음"·"미참석으로 인한 문제의 책임은 입찰자" → 미참석 결과가 '참가 불가'일 때만 인정
#   ② "설명회 참석은 필수가 아니며"·"참석 여부는 입찰 참여의 필수 조건이 아니" → 필수/의무 뒤 부정어 배제
#   ③ 과업 내용의 "보고회·설명회·공청회 … 참여하여 보고하여야"(계약상대자 의무) → 과업 문맥 배제
#   ④ "참석자(현지 여행사)를 통한 … 유치가 가능"·"참석자 사전통보 … 3명까지 참석 가능" → '가능'은 참가·입찰 가능에 한정
BRIEF_GATE_PAT = [
    re.compile(r"(?:설명회|현장\s*설명)[^가-힣]{0,10}(?:에|를|는|,)?[^\n]{0,80}?"
               r"(?:참석|참여|참가)(?:한|업체|자|기업)[^\n]{0,40}"
               r"(?:한하여|한함|에\s*한|자격|부여|(?:참가|참여|입찰|제출)(?:이|가|을|를)?\s*가능)"),
    re.compile(r"(?:미\s*참석|불참|참석하지\s*(?:아니|않)[가-힣]*|참여하지\s*(?:아니|않)[가-힣]*)"
               r"[^\n]{0,50}(?:제외|불가|허용되지|접수하지|"
               r"(?:참가|참여|입찰|제출|접수)(?:할\s*수|이|가|은|는|을|를)?\s*(?:없|않|불가))"),
    re.compile(r"(?:설명회|현장\s*설명)[^\n]{0,60}(?:참석|참여|참가)[^\n]{0,25}?"
               r"(?:필수|의무|하여야|해야)(?![^\n]{0,12}(?:아니|않|사항은\s*아))"),
    re.compile(r"(?:설명회|현장\s*설명)(?:에|를)?\s*(?:참석|참여|참가)한\s*자"),
]
# 미참석의 결과가 '불이익·책임'에 그치거나(참가 제한 아님), 설명회가 계약상대자의 과업(보고회·공청회)이거나,
# "참석하지 아니한 업체도 … 될 수 있다"처럼 미참석을 허용하는 창은 참가자격 강제가 아니다.
BRIEF_GATE_NEG = re.compile(
    r"불이익|책임|귀책|보고회|공청회|간담회|의견\s*청취|계약상대자|과업\s*수행|사업\s*책임자|"
    r"(?:업체|자|기업)도\s*[^\n]{0,20}(?:될\s*수\s*있|가능)|필수(?:가|는|사항은)?\s*아니|필수\s*조건이\s*아니|"
    r"관계\s*없이|참가\s*제한\s*없|무관하게|사전\s*통보|관광\s*설명회|유치|인센티브|홍보\s*방안|위임장|지참|여부|숙지|열람")
BRIEF_OMIT = re.compile(r"생략|없음|없습니다|(?:실시|개최|진행|운영)(?:하지|는\s*하지|를\s*하지)?\s*(?:않|아니)|"
                        r"갈음|대체|미실시|미개최|해당\s*없|별도로\s*(?:실시|개최|진행)하지")
BRIEF_SCHED_NEG = re.compile(r"결과\s*설명회|성과|중간보고|착수보고|워크숍|피칭|채용|주관\s*사업설명회")
# [추가] 개최일 탐지용 표제어 — "5. 제안요청서 설명: 실시함 — 일시 2026. 2. 6."처럼 '설명회' 없이 '제안요청서 설명'으로만
#        적는 공고(dev 141, 정답 1)를 놓쳤다. '설명서'·'설명 자료'는 제외.
BRIEF_SCHED_HIT = re.compile(BRIEF_HIT.pattern + r"|제안\s*요청서?\s*설명(?!서|\s*자료)")
PROP_DUE_PAT = re.compile(
    r"제안서[^\n]{0,30}?(?:접수|제출)[^\n]{0,10}?(?:기간|일시|마감|기한)?|기술제안서[^\n]{0,20}접수|"
    r"제안서\s*및\s*가격입찰서\s*제출|접수\s*마감")


def detect_brief_gate(rec: Dict[str, Any]) -> Tuple[bool, str]:
    """설명회 참석을 입찰·제안 참가자격으로 강제하는지."""
    t = flat_text(rec)
    for m in BRIEF_HIT.finditer(t):
        w = t[max(0, m.start() - 120): m.end() + 220]
        if BRIEF_NEG.search(w):
            continue
        for p in BRIEF_GATE_PAT:
            mm = p.search(w)
            if not mm:
                continue
            # 부정 문맥은 강제 문구 자체와 그 앞뒤 짧은 구간에서만 본다(창 전체를 보면 인접 조항의
            # "불이익"·"위임장"이 진짜 강제 문구까지 지운다 — 자가라벨 대조에서 TP 6건 손실).
            zone = w[max(0, mm.start() - 15): mm.end() + 12]
            if BRIEF_GATE_NEG.search(zone):
                continue
            return True, t[max(0, m.start() - 60): m.end() + 180]
    return False, ""


def _dates_in(s: str) -> List[Tuple[int, int, int]]:
    out: List[Tuple[int, int, int]] = []
    for m in _DATE_PAT.finditer(s):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            out.append((y, mo, d))
    return out


def detect_brief_schedule(rec: Dict[str, Any]) -> Tuple[Optional[Tuple[int, int, int]],
                                                        Optional[Tuple[int, int, int]]]:
    """(설명회 개최일, 설명회 이후 최초 제안서 제출마감일)."""
    t = flat_text(rec)
    brief: Optional[Tuple[int, int, int]] = None
    for m in BRIEF_SCHED_HIT.finditer(t):
        w = t[max(0, m.start() - 80): m.end() + 160]
        if BRIEF_SCHED_NEG.search(w):
            continue
        # [추가] "사업설명회 : 생략(제안요청서로 갈음)"·"설명회: 없음"처럼 설명회를 열지 않는 공고는 개최일이
        #        없다 — 이전에는 다음 줄의 등록·마감 날짜를 개최일로 오인해 v23 을 발화했다(dev 121 FP,
        #        자가라벨 대조 v23 FP 27건 중 23건이 이 유형).
        if BRIEF_OMIT.search(t[m.end(): m.end() + 40]):
            continue
        ds = _dates_in(t[m.end(): m.end() + 160])
        if ds:
            c = min(ds, key=ordinal_day)
            if brief is None or ordinal_day(c) < ordinal_day(brief):
                brief = c
    if brief is None:
        return None, None
    cands: List[Tuple[int, int, int]] = []
    for m in PROP_DUE_PAT.finditer(t):
        ds = _dates_in(t[m.end(): m.end() + 140])
        if ds:
            cands.append(max(ds, key=ordinal_day))
    later = [c for c in cands if days_between(brief, c) >= 1]
    return brief, (min(later, key=ordinal_day) if later else None)





# ===================================================================================
# 7. LLM 사실 추출 — 스키마 · 프롬프트
# ===================================================================================
SIZE_ENUM = ["없음", "중소기업", "소기업소상공인", "기타"]
REGION_ENUM = ["없음", "광역", "기초", "복수광역", "기타"]

FACT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "inst", "inst_q", "perf", "perf_amt", "perf_org", "perf_q",
        "reg", "reg_lv", "reg_q", "size", "size_q", "dpc", "dpc_q", "cmp",
        "model", "model_q", "pled", "pled_q", "brief", "brief_q",
        "sw", "swlim", "exc", "share", "share_q", "mism", "mism_q",
        "brief_date", "prop_due",
    ],
    "properties": {
        # 참가자격 — 기관/실적
        "inst": {"type": "integer", "enum": [0, 1]},
        "inst_q": {"type": ["string", "null"], "maxLength": 300},
        "perf": {"type": "integer", "enum": [0, 1]},
        "perf_amt": {"type": "integer", "minimum": 0},
        "perf_org": {"type": "integer", "enum": [0, 1]},
        "perf_q": {"type": ["string", "null"], "maxLength": 300},
        # 참가자격 — 지역
        "reg": {"type": "integer", "enum": [0, 1]},
        "reg_lv": {"type": "string", "enum": REGION_ENUM},
        "reg_q": {"type": ["string", "null"], "maxLength": 300},
        # 참가자격 — 기업규모 / 직접생산
        "size": {"type": "string", "enum": SIZE_ENUM},
        "size_q": {"type": ["string", "null"], "maxLength": 300},
        "dpc": {"type": "integer", "enum": [0, 1]},
        "dpc_q": {"type": ["string", "null"], "maxLength": 300},
        "cmp": {"type": "integer", "enum": [0, 1]},
        # 규격 · 확약서 · 설명회
        "model": {"type": "integer", "enum": [0, 1]},
        "model_q": {"type": ["string", "null"], "maxLength": 300},
        "pled": {"type": "integer", "enum": [0, 1]},
        "pled_q": {"type": ["string", "null"], "maxLength": 300},
        "brief": {"type": "integer", "enum": [0, 1]},
        "brief_q": {"type": ["string", "null"], "maxLength": 300},
        # 소프트웨어 · 예외 · 공동수급 · 메타대조
        "sw": {"type": "integer", "enum": [0, 1]},
        "swlim": {"type": "integer", "enum": [0, 1]},
        "exc": {"type": "integer", "enum": [0, 1]},
        "share": {"type": "number"},
        "share_q": {"type": ["string", "null"], "maxLength": 300},
        "mism": {"type": "integer", "enum": [0, 1]},
        "mism_q": {"type": ["string", "null"], "maxLength": 300},
        # 협상계약 일정
        "brief_date": {"type": ["string", "null"], "maxLength": 20},
        "prop_due": {"type": ["string", "null"], "maxLength": 20},
    },
}

SYSTEM_PROMPT = """당신은 대한민국 공공조달 입찰공고를 검토하는 계약심사 전문가다.
주어진 입찰공고 발췌를 읽고 **사실만** JSON으로 추출한다. 위법 여부는 판단하지 않는다.

공통 원칙
- 각 항목의 근거 인용문(_q)은 **원문에서 글자 그대로 복사**한 1~2문장(120자 이내)이어야 한다.
  요약·의역·번역·줄임표 추가 금지. 해당 사실이 없으면 null.
- '참가자격'으로 요구된 것만 인정한다. 제출서류 목록·평가기준·일반 안내문구에만 등장하는 것은
  참가자격 요건이 아니다.
- 청렴계약·부정당업자·전자입찰 등록·국세완납 같은 모든 공고에 붙는 상투적 문구는 제한이 아니다.

필드 정의
1) inst : 특정 기관·단체·법인 유형만 참가할 수 있게 한정했으면 1.
   예) 대학·산학협력단·연구기관·협회·조합·재단·특정 인증기관·특정 단체 회원만 참가 가능.
   업종/면허 등록(나라장터 업종코드), 지역, 기업규모는 여기에 해당하지 않는다. inst_q = 그 문장.
   0인 경우: 관계 법령이 의무화한 허가·등록·신고 자격(폐기물처리업 허가, 정보통신공사업·문화재수리업 등록,
   경비업 허가, 여행업 등록 등), 비영리법인·사회적기업·장애인기업을 '우대·가점'만 하는 경우,
   해당 단체·법인'도' 참가할 수 있다고 포함을 밝힌 경우.
2) perf : 과거 납품·수행 '실적'을 참가자격으로 요구하면 1.
   perf_amt = 요구 실적금액을 **원 단위 정수**로 (예: '3억원 이상' → 300000000, '1억 원' → 100000000).
              금액이 아니라 건수·규모만 요구하면 0.
   perf_org : 실적의 **발주처·납품처를 특정 기관으로 한정**하면 1
              (예: '국가기관이 발주한', '공공기관에서 발주한', '지방자치단체 발주', '대학병원에 납품한').
              발주처 제한 없이 유사 실적만 요구하면 0.
   perf_q = 실적 요구 문장.
3) reg : 본점·주된 영업소 소재지(지역)를 참가자격으로 제한하면 1.
   reg_lv : '광역'=특별시/광역시/도/특별자치도 1곳 단위,
            '기초'=시·군·구·읍면동 등 기초자치단체 단위로 좁힘,
            '복수광역'=둘 이상의 시·도(인접 시·도 포함)로 확대,
            '없음'=지역제한 없음.
   reg_q = 지역제한 문장.
4) size : 참가자격으로 요구한 **기업규모**. 참가자격 조항의 문장 하나를 근거로 정한다.
   '중소기업'  = 중소기업자 전체(중기업 포함). '중소기업확인서를 소지한 자', '중기업 또는 소기업이거나 소상공인',
                 그리고 **'중소기업(자) 또는 소상공인', '중·소기업·소상공인 확인서'처럼 중소기업과 소상공인을
                 병기**한 문장도 중기업이 포함되므로 '중소기업'이다.
   '소기업소상공인' = 자격 문장이 **'소기업'만** 적어 중기업을 배제한 경우
                 ('소기업 또는 소상공인', '소기업·소상공인 확인서를 소지한 자', 벤처·창업기업 포함).
                 자격 문장에 '중소기업'이 규모 요건으로 함께 쓰였으면 '소기업소상공인'이 아니라 '중소기업'이다
                 (단 '「중소기업기본법」 제2조제2항에 따른 소기업'처럼 법령 이름 안의 '중소기업'은 무시).
   '없음'      = 기업규모 요건이 참가자격에 전혀 없음. 제출서류 목록의 '소기업·소상공인 확인서 1부',
                 적격심사 배점표의 '중기업/소기업(소상공인)' 구분란, 확인서 발급 방법 안내문은 참가자격이 아니므로 '없음'.
   '기타'      = 그 밖(대기업 허용 명시 등).
   size_q = 기업규모 요구 문장(참가자격 조항의 문장, 안내문·제출서류 줄이 아니어야 한다).
5) dpc : 「판로지원법」에 따른 **직접생산확인증명서 보유**를 참가자격으로 요구하면 1. dpc_q = 그 문장.
6) cmp : 이 조달 대상이 **중소기업자간 경쟁제품**(중소벤처기업부 지정 품목)에 해당하면 1.
   아래 '경쟁제품 후보' 목록에 조달 대상과 같은 품목이 있으면 1로 본다.
   본문에 '중소기업자간 경쟁제품', '직접생산확인증명서', 경쟁제품 세부품명번호가 나오면 1.
7) model : 규격서·과업지시서·공고문이 **특정 제조사·상표·모델명**을 지정하면 1
   ('동등 이상' 참고 표기 포함). 일반 규격·성능 수치만 있으면 0. model_q = 그 문장.
8) pled : 제조사·공급사의 **물품공급(기술지원·A/S) 확약서·공급증명원**을 **입찰 참가 시점**에 요구하면 1 —
   즉 '입찰서(제안서) 제출 마감(전일)까지 보유·발급·제출', '미보유 시 입찰 대상 제외', 또는 입찰 참가
   제출서류로 확약서를 요구한 경우. 마감 전 보유를 요구하면서 제출만 계약 시로 미룬 경우도 1.
   0인 경우: 마감 전 보유 요구 없이 '계약 체결 시·낙찰 후·적격심사 시·납품(검수) 시'에만 제출하도록 한 경우,
   청렴서약·입찰보증금 납부이행 확약처럼 물품공급 확약이 아닌 것. pled_q = 시점이 드러난 문장.
9) brief : **현장설명회·사업설명회 참석을 입찰(제안서 제출) 자격 요건으로** 삼으면 1
   (미참석 업체 참가 불가/제안서 미접수). 설명회를 단순히 개최한다고만 하면 0. brief_q = 그 문장.
10) sw : 소프트웨어 개발·정보시스템 구축/운영·유지관리 등 **소프트웨어사업**이면 1.
    swlim : 공고문·제안요청서에 **대기업(중견기업) 참여제한 하한 제도 적용 여부**를 명시했으면 1
            ('대기업인 소프트웨어사업자 참여제한 적용/미적용', '대기업 참여 가능/제한' 등).
11) exc : 「판로지원법 시행령」 제2조의3의 **예외 사유를 공고문에 명시**했으면 1
    (비영리법인 참여 필요, 다른 법령의 우선구매, 특정 기술·성능 필요, 국제입찰 대상 등).
12) share : 공동수급체 **구성원별 최소지분율(%)**. 명시가 없으면 -1. share_q = 그 문장.
13) mism : 공고문 본문이 밝힌 값과 아래 '나라장터 입력값(메타)'이 **서로 다르면** 1.
    비교 대상은 ①예산/추정가격 ②계약방법(일반경쟁·제한경쟁·수의계약) ③지역제한 대상 지역 ④업종·면허.
    메타가 '미기재/None'이면 불일치로 보지 않는다. mism_q = 공고문 쪽 문장.
14) brief_date / prop_due : 협상에 의한 계약일 때 **제안요청설명회(현장설명회) 일자**와
    **제안서 제출 마감일자**를 'YYYY-MM-DD'로. 없으면 null. 개찰일이 아니라 제안서 마감일이다.

반드시 위 스키마의 JSON 객체 하나만 출력한다."""


def format_meta_block(rec: Dict[str, Any]) -> str:
    m = rec.get("meta", {})
    lines = []
    for k in META_FIELDS:
        if k in m:
            v = m[k]
            if v is None or str(v).strip() == "":
                continue
            lines.append(f"- {k}: {v}")
    return "\n".join(lines)


def build_user_prompt(rec: Dict[str, Any], pre: Dict[str, Any]) -> str:
    nt = notice_text(rec)
    st = spec_text(rec)
    parts: List[str] = []
    parts.append("## 나라장터 입력값(메타)\n" + format_meta_block(rec))

    hint = []
    est = pre["est"]
    hint.append(f"추정가격 {est:,}원 / 배정예산 {pre['budget']:,}원 / "
                f"{'지방계약법' if pre['local'] else '국가계약법'}")
    if pre["share_hint"]:
        hint.append("본문에서 찾은 지분율(%): " + ", ".join(str(x) for x in pre["share_hint"][:5]))
    if pre["comp_codes"]:
        hint.append("본문 세부품명번호 중 경쟁제품 목록과 일치: " + ", ".join(pre["comp_codes"][:4]))
    if pre.get("comp_excluded"):
        hint.append("주의: 아래 품목은 중기부고시 경쟁제품이나 특이사항의 추정가격 조건을 벗어나 "
                    "이 건에서는 경쟁제품이 아니다 → " + ", ".join(pre["comp_excluded"][:3]))
    if pre["comp_names"]:
        hint.append("경쟁제품 후보(유사 품목명): " + ", ".join(pre["comp_names"]))
    if pre["amount_mismatch"]:
        hint.append("주의: 본문 추정가격 표기가 메타와 달라 보인다 → " + clip(pre["amount_mismatch_q"], 120))
    if pre.get("method_mismatch"):
        hint.append("주의: 본문 계약·입찰 방법 표기가 메타의 계약방법과 달라 보인다 → "
                    + clip(pre.get("method_mismatch_q") or "", 120))
    if pre.get("biz_mismatch"):
        hint.append("주의: 본문 업종코드가 메타의 면허업종제한목록과 겹치지 않는다 → "
                    + clip(pre.get("biz_mismatch_q") or "", 120))
    if pre.get("size_det"):
        hint.append(f"규칙검출: 참가자격의 기업규모 제한 = {pre['size_det']} → "
                    + clip(pre.get("size_det_q") or "", 120))
    if pre.get("comp_panro9"):
        hint.append("규칙검출: 판로지원법 제9조(직접생산확인) 인용 → 중소기업자간 경쟁제품 공고일 가능성")
    if pre.get("pledge_strong"):
        hint.append("규칙검출: 확약서·공급증명원을 입찰서 제출 마감 전까지 보유·제출하도록 강제 → "
                    + clip(pre.get("pledge_strong_q") or "", 120))
    if pre.get("reg_det"):
        lv = "기초(시·군·구)" if pre.get("reg_basic") else ("둘 이상 광역" if pre.get("reg_multi") else "광역")
        hint.append(f"규칙검출: 지역제한 있음 · 단위={lv} → " + clip(pre.get("reg_quote") or "", 120))
    if pre.get("sw_det"):
        hint.append("규칙검출: 소프트웨어사업(소프트웨어사업자 자격 요구)"
                    + ("  · 대기업 참여제한 명시 있음" if pre.get("sw_limit") else "  · 대기업 참여제한 언급 없음"))
    if pre.get("pledge_det"):
        hint.append("규칙검출: 물품공급·기술지원 확약서 관련 문구 → " + clip(pre.get("pledge_quote") or "", 120))
    if pre.get("brief_gate"):
        hint.append("규칙검출: 설명회 참석을 참가자격으로 제한하는 문구 → " + clip(pre.get("brief_gate_quote") or "", 120))
    if pre.get("brief_day"):
        bd = pre["brief_day"]
        hint.append("규칙검출: 설명회 개최일 %04d-%02d-%02d" % bd)
    if pre.get("prop_day"):
        pd = pre["prop_day"]
        hint.append("규칙검출: 제안서 제출마감일 %04d-%02d-%02d" % pd)
    parts.append("## 참고 힌트\n" + "\n".join(f"- {h}" for h in hint))

    parts.append("## [공고 개요]\n" + notice_head(nt, 1200))
    qual = pre["qual"]
    parts.append("## [입찰참가자격]\n" + (qual if qual else "(참가자격 조항을 찾지 못함)"))

    extra = []
    joint = extract_around(nt, r"공동\s*수급|공동\s*계약|공동\s*도급|지분율|출자비율", 380, 2, 900)
    if joint:
        extra.append("### 공동수급\n" + joint)
    brief = extract_around(nt, r"현장\s*설명|사업\s*설명회|제안요청\s*설명|설명회", 380, 3, 1000)
    if brief:
        extra.append("### 설명회\n" + brief)
    due = extract_around(nt, r"제안서\s*(?:제출|접수)|입찰서\s*제출|제출\s*마감|접수\s*마감|개찰", 260, 3, 800)
    if due:
        extra.append("### 제출·마감 일정\n" + due)
    pledge = extract_around(nt + "\n" + st, r"확약서|공급확약|기술지원|공급\s*협약", 320, 2, 700)
    if pledge:
        extra.append("### 확약서\n" + pledge)
    swlim = extract_around(nt + "\n" + st, r"대기업|중견기업|소프트웨어사업자|상호출자제한", 300, 2, 700)
    if swlim:
        extra.append("### 대기업 참여\n" + swlim)
    model = extract_around(
        st + "\n" + nt,
        r"모델\s*명|모델명|제조사|제조업체|상표|브랜드|동등\s*이상|규격\s*[:：]|품명\s*[:：]", 300, 3, 1100)
    if model:
        extra.append("### 규격·모델\n" + model)
    if extra:
        parts.append("## [관련 발췌]\n" + "\n".join(extra))

    return "\n\n".join(parts)


def build_messages(rec: Dict[str, Any], pre: Dict[str, Any]) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(rec, pre)},
    ]


# 나라장터 메타 '조항호내용'이 경쟁제품 근거조항을 직접 밝히는 경우가 있다.
# 같은 줄의 다른 법률(「여성기업 지원에 관한 법률」제9조 등)로 건너뛰지 않도록 인용부호를 경계로 둔다 (dev 04 과탐).
_RE_PANRO_DPC = re.compile(r"판로지원에\s*관한\s*법률\s*[」｣]?[^\n「」｢｣]{0,60}제\s*9\s*조")
_RE_JOH_CMP = re.compile(r"지정\s*공고한\s*물품|지정.?고시한\s*제품|중기간\s*경쟁제품|경쟁제품")

# --- v24 공고서 ↔ 나라장터 입력값 대조 -----------------------------------------
# 항목표 비고가 지목한 축은 예산·계약방법·지역제한·업종이다. dev 200건 실측 결과
# 금액 축(F1 0.286) 단독보다 계약방법·업종코드 축을 더한 쪽이 낫다(F1 0.421).
_BID_METHODS = ("일반경쟁", "제한경쟁", "지명경쟁", "수의계약")
_RE_METHOD_LINE = re.compile(
    r"(?:입찰\s*방법|계약\s*방법|입찰\s*및\s*계약\s*방[식법]|입찰방식)[^\n]{0,60}")
_RE_BIZ_CODE = re.compile(
    r"업종\s*코드\s*[:：]?\s*(\d{4})|\(\s*업종코드\s*[:：]?\s*(\d{4})\s*\)"
    r"|\[\s*업종코드\s*(\d{4})\s*\]")


def detect_method_mismatch(rec: Dict[str, Any]) -> Tuple[bool, str]:
    """메타 계약방법과 공고문이 밝힌 입찰방법이 어긋나는지."""
    meta = meta_str(rec, "계약방법").strip()
    if meta not in _BID_METHODS:
        return False, ""
    text_all = notice_text(rec)
    # [2026-09-19c] 기존 early return(첫 300자에 메타 방법이 있으면 면제)은
    # 제목 "(일반경쟁·1억원미만)" 처럼 나라장터 값 그대로 복사된 텍스트가 진짜 불일치를
    # 가리는 FN 12건의 원인이었다. → early return 제거하고, 방법줄만으로 판정한다.
    found, quote = set(), ""
    for m in _RE_METHOD_LINE.finditer(text_all):
        line = m.group(0)
        for k in _BID_METHODS:
            if k in line or (k == "수의계약" and "수의" in line):   # "소액수의견적" 도 수의계약 표기다(dev FP 148·194)
                found.add(k)
                quote = quote or line.strip()[:300]
    if found and meta not in found:
        return True, quote                        # 근거는 공고문 원문 그대로여야 한다
    return False, ""


def detect_biz_code_mismatch(rec: Dict[str, Any]) -> Tuple[bool, str]:
    """메타 면허업종제한목록의 업종코드와 공고문 업종코드가 하나도 겹치지 않는지."""
    meta_codes = set(re.findall(r"(\d{4})", meta_str(rec, "면허업종제한목록")))
    if not meta_codes:
        return False, ""
    text = notice_text(rec)
    doc_codes, quote = set(), ""
    for m in _RE_BIZ_CODE.finditer(text):
        doc_codes.add(next(g for g in m.groups() if g))
        if not quote:
            a = text.rfind("\n", 0, m.start()) + 1
            b = text.find("\n", m.end())
            quote = text[a: b if b > 0 else len(text)].strip()[:300]
    if doc_codes and not (doc_codes & meta_codes):
        return True, quote                        # 근거는 공고문 원문 그대로여야 한다
    return False, ""


# --- v24 공고 제목 가격범주 불일치 검출기 ----------------------------------------
# 공고 제목(첫 400자)에 "N억원미만" 또는 "N천만원미만" 이 명시되어 있는데
# 나라장터 입찰추정가격이 그 상한을 초과하면 메타 입력값 불일치(v24)로 판정한다.
# dev200 실측: 이 규칙으로 DEV-055(FN) → TP 전환, FP 추가 0건.
_RE_TITLE_PRICE_LIMIT = re.compile(r"(\d+)\s*억\s*원?\s*미만")
_RE_TITLE_PRICE_LIMIT_CHUN = re.compile(r"(\d+)\s*천\s*만\s*원?\s*미만")


def detect_title_price_category_mismatch(rec: Dict[str, Any], est: int) -> Tuple[bool, str]:
    """공고 제목이 명시한 가격 상한(N억원미만)과 나라장터 추정가격이 어긋나는지."""
    if est <= 0:
        return False, ""
    title_area = notice_text(rec)[:400]
    m = _RE_TITLE_PRICE_LIMIT.search(title_area)
    if not m:
        m = _RE_TITLE_PRICE_LIMIT_CHUN.search(title_area)
        if m:
            threshold = int(m.group(1)) * 10_000_000
        else:
            return False, ""
    else:
        threshold = int(m.group(1)) * 100_000_000
    if est > threshold:
        start = max(0, m.start() - 20)
        quote = title_area[start: m.end() + 20].strip()[:200]
        return True, quote
    return False, ""


def precompute(rec: Dict[str, Any], comp_codes: Dict[str, str],
               comp_names: List[Tuple[str, str]]) -> Dict[str, Any]:
    """LLM 호출 전 결정적으로 계산되는 값."""
    nt = notice_text(rec)
    ft = full_text(rec)
    est = to_int(rec["meta"].get("입찰추정가격"))
    budget = to_int(rec["meta"].get("배정예산금액"))
    if est <= 0 and budget > 0:
        est = int(budget / 1.1)
    if budget <= 0 and est > 0:
        budget = int(est * 1.1)

    # [변경] "세부품명번호 10자리 : 7811189902"·"G2B분류번호 10자리(5512170601, 현수막"·"세부물품번호:"·"세부제품번호:"·
    #        "[8111200202]"·메타 "현수막[5512170601]" 을 놓쳐 경쟁제품(레미콘·조경석·현수막·통학운송)을 일반제품으로 오판(무라벨 20k v12 FP 9건).
    codes = set(re.findall(r"(?:세부\s*(?:품명|물품|제품)\s*번호|G2B\s*분류번호|물품분류번호|품명번호)(?:\s*10\s*자리)?[^0-9]{0,25}(\d{10})", ft))
    # [추가] "기타행사기획및대행서비스(8014199001)" 처럼 품명 바로 뒤 괄호에 적힌 10자리 코드(dev 053·169)
    codes |= set(re.findall(r"[가-힣A-Za-z]\s*[\(\[](\d{10})[\)\]]", ft))
    codes |= set(re.findall(r"\d{10}", str(rec["meta"].get("세부품명번호목록") or "")))
    # [추가] "(세부품명:7811189902)" 처럼 '번호' 없이 콜론 뒤 10자리, 표 셀에 코드만 적힌 줄(무라벨 20k v12 FN 2건).
    #        어차피 고시 목록(615개 코드)에 있는 번호만 인정되므로 전화번호 등 잡음은 걸러진다.
    codes |= set(re.findall(r"세부\s*품명\s*[:：]\s*(\d{10})", ft))
    codes |= set(re.findall(r"(?:^|\n)\s*(\d{10})\s*(?=\n|$)", ft))
    notes = competition_notes()
    all_hits = sorted(c for c in codes if c in comp_codes)
    hit_codes = [c for c in all_hits if note_amount_ok(notes.get(c, ""), est)]
    excluded = [c for c in all_hits if c not in hit_codes]

    title = guess_title(nt)
    cand = competition_candidates(title, comp_names, topk=5)

    mism, mism_q = detect_meta_amount_mismatch(rec)
    meth_mm, meth_q = detect_method_mismatch(rec)
    biz_mm, biz_q = detect_biz_code_mismatch(rec)
    title_price_mm, title_price_q = detect_title_price_category_mismatch(rec, est)
    exc_doc = any("예외" in str(d.get("type") or "") for d in rec.get("docs", []))
    reg_d = detect_region(rec)
    sw_d = detect_software(rec)
    pled_hit, pled_q = detect_pledge(rec)
    pled_s, pled_sq = detect_pledge_strong(rec)
    brief_hit, brief_q = detect_brief_gate(rec)
    brief_day, prop_day = detect_brief_schedule(rec)
    out = {
        "pledge_det": pled_hit,
        "pledge_quote": pled_q,
        "pledge_strong": pled_s,
        "pledge_strong_q": pled_sq,
        "brief_gate": brief_hit,
        "brief_gate_quote": brief_q,
        "brief_day": brief_day,
        "prop_day": prop_day,
        "exception_doc": exc_doc,
        "reg_det": reg_d["reg"],
        "reg_basic": reg_d["basic"],
        "reg_multi": reg_d["multi"],
        "reg_quote": reg_d["quote"],
        "sw_det": sw_d["sw"],
        "sw_limit": sw_d["limit"],
        "sw_quote": sw_d["quote"],
        "est": est,
        "budget": budget,
        "local": is_local_law(rec),
        "band": price_band(est),
        "qual": extract_qualification(nt),
        # 부재탐지(v16·v18) 전제: 진짜 자격 조항을 읽었고, 규모 확인서 요구도 없어야 한다.
        "qual_found": qualification_section_found(nt),
        "size_doc_hint": bool(_RE_SIZE_DOC.search(nt)),
        "share_hint": parse_share_ratios(nt),
        "comp_codes": [f"{c}({comp_codes[c]})" for c in hit_codes],
        "comp_code_hit": bool(hit_codes),
        # 세부품명번호가 표기되지 않는 서비스성 경쟁제품 보완 신호
        "comp_name_hit": competition_best_sim(title, comp_names) >= 0.5,
        "comp_joh": bool(_RE_JOH_CMP.search(meta_str(rec, "조항호내용"))),
        "comp_panro9": bool(_RE_PANRO_DPC.search(ft)),
        # [추가] 세부품명번호 없이 "기타행사기획 및 대행서비스 부문의 직접생산확인증명서" 처럼
        #        경쟁제품 품명을 문장으로 적은 공고(dev 077). 직접생산 문장 안에서만 찾는다.
        "comp_text_hit": competition_name_in_dpc(ft, comp_names),
        "is_goods": "물품" in meta_str(rec, "업무구분"),
        # [추가] 세부품명번호가 없는 용역 경쟁제품 보완 신호 2종 (dev 용역 122건 대조).
        #  · 면허업종제한 '행사대행업(9901)' = 기타행사기획및대행서비스(8014199001, 추정가격 10억 미만에 한함).
        #    dev 양성 061·062·069·075 모두 해당, 10억 초과 25(11.4억)·141(32.7억)은 고시 특이사항으로 제외된다.
        #  · 제목의 "…시스템/체계 유지보수" = 정보시스템유지관리서비스(8111189901)·승강기유지보수서비스(7215401001).
        #    [조사기록 2026-09-10] dev200 에서 comp_event 가 유일 신호인 건은 4건(039·062·069·191)이고
        #    062·069 는 정답이 v10/v11/v13(경쟁제품 분기)이라 이 신호가 반드시 필요하다. 039 만 정답이 v18
        #    (일반제품 분기)인데, 039 공고문은 "입찰방법: 제한경쟁(소기업·소상공인)"을 개요에 명시하고도 정답이
        #    v18(=소기업 제한 부재)이라 정답끼리 모순이다 → 039 한 건 때문에 신호를 끄지 않는다.
        "comp_event": bool(re.search(r"9901|행사\s*대행업", meta_str(rec, "면허업종제한목록")))
                      and 0 < est < 1_000_000_000,
        "comp_sysmaint": bool(re.search(r"(?:시스템|체계|승강기)[^\n]{0,20}유지\s*(?:보수|관리)", title or "")),
        # 나라장터 조항호내용이 '특수기술 용역' 예외를 명시하면 v16·v18 부재탐지 면제
        "exc_joh": bool(re.search(r"특수한\s*기술이\s*요구", meta_str(rec, "조항호내용"))),
        "comp_excluded": [f"{c}({comp_codes[c]}·{notes.get(c, '')})" for c in excluded],
        "comp_names": cand,
        "amount_mismatch": mism,
        "amount_mismatch_q": mism_q,
        "method_mismatch": meth_mm,
        "method_mismatch_q": meth_q,
        "biz_mismatch": biz_mm,
        "biz_mismatch_q": biz_q,
        "title_price_mismatch": title_price_mm,
        "title_price_mismatch_q": title_price_q,
        "title": title,
    }
    out.update(backstop_facts(out["qual"], ft))
    return out



# ===================================================================================
# 7-b. 결정론적 백스톱 — LLM 전용 항목의 하한선
# ===================================================================================
# v5~v7(지역)·v20 처럼 잘 맞는 항목은 모두 `pre(결정론) OR f(LLM)` 앙상블이고,
# v1·v4·v9 처럼 F1 0 인 항목은 LLM 단독이었다. 그 비대칭을 메운다.
# 원칙: 재현율이 아니라 **정밀도**를 우선한다. OR 로 합쳐지므로 오탐은 그대로 손실이다.
# 아래 패턴은 dev 200건에서 정밀도 0.6~1.0 을 확인한 것만 남겼다.

# v1 참가자격 특정기관 제한 — dev 200건 TP5/FP0 (P 1.000 · R 0.714)
_INST_TERM = (r"산학협력단|연구기관|연구원|협회|조합|재단법인|사단법인|비영리법인"
              r"|대학|기관")
_RE_INST = re.compile(
    rf"(?:{_INST_TERM})[^\n]{{0,30}}만\s*(?:참여|참가|입찰|응찰)"      # "…대학교만 참여 가능"
    r"|\d+\s*명\s*이상의[^\n]{0,30}보유(?:한|하고)"                   # 인력 보유 요구
    r"|전국[^\n]{0,40}(?:있는|보유한)\s*업체")                         # 전국 거점 보유 요구

# [추가] 규모 확인서 문장("중소기업·소상공인 확인서 및 비영리법인 확인서를 보유한 업체만", "소기업·소상공인 확인서(용도: 공공기관
#        입찰용)를 소지한 자만")과 실적·소재지 문장("전국 규모의 축제 수행 실적이 있는 업체", "본점소재지가 전국에 있는 업체")은
#        기관 유형 한정이 아니다(무라벨 20k v1 FP 4건; dev 양성 5건에는 이 단어들이 없다).
_RE_INST_BOILER = re.compile(r"경우에만|발주\s*기관|수요\s*기관|계약\s*기관|협동조합|적격조합|확인서|실적|소재지[가는]?\s*전국")
# [추가] 조합 추천 업체로 참가를 한정하는 진짜 v1 위반 문구. 보일러플레이트 억제의 예외로 쓴다.
#   예) "…협동조합으로부터 추천 받은 아래 5개 업체만 입찰이 가능합니다", "조합에서 추천받은 5개 업체만 입찰서 제출 가능"
_RE_INST_RECOMMEND = re.compile(r"추천\s*받?은?[^\n]{0,25}(?:업체|사업자|자)\s*만|추천\s*받?은?[^\n]{0,15}\d+\s*개[^\n]{0,10}업체")

# v2·v3·v8 실적 제한 — 근거법상 "최근 N년 실적" 이 전형적 표현이다.
_RE_PERF_DET = re.compile(
    r"최근\s*\d+\s*년[^\n]{0,80}실적"
    r"|(?:납품|시공|수행|이행|계약|공급)\s*실적[^\n]{0,60}(?:있는|보유|이상|갖춘|업체|자)"
    # [추가 2026-09-16] 자가라벨 프록시 v2 FN(자격블록 안) 4유형: "…용역을 수행한 실적(…완성된 실적만 인정)",
    #   "관련 분야 3회 이상 실적 보유 기관", "컨설팅 유경험 업체", "최근 3년 이내 <긴 수식어 60자+> 운영·관리 실적"
    r"|(?:납품|시공|수행|이행|계약|공급|운영|관리)한\s*실적"
    r"|\d+\s*(?:회|건)\s*이상[^\n]{0,20}실적"
    r"|실적\s*(?:을\s*)?보유"
    r"|유경험\s*(?:업체|자|기업|법인|기관)")

# 실적 줄 판별 보조: 요구 표지가 있으면 채택, 없고 서류·평가 어휘만 있으면 제출목록/평가항목으로 본다.
_RE_PERF_REQ = re.compile(r"이상|보유|있는\s*(?:업체|자|단체|법인|기업)|이어야|여야|제한|자로서|업체로서|경험이|경험을|유경험|만\s*인정")
_RE_PERF_DOC = re.compile(r"증명서|서류|각\s*\d*\s*부|평가|배점|현황|제출|첨부|양식|서식|해당되는\s*경우|인정합니다")


def _line_of(text: str, pos_s: int, pos_e: int) -> str:
    s0 = text.rfind("\n", 0, pos_s) + 1
    e0 = text.find("\n", pos_e)
    return text[s0: e0 if e0 >= 0 else len(text)].strip()


# [추가] "납품실적으로 제한하지는 않으나 ... 평가항목 및 배점은 있으며" 같은 명시적 부정문과
#        "공익활동실적·봉사실적"(계약이행 실적이 아님)은 실적 제한이 아니다(무라벨 20k v8 FP).
_RE_PERF_NEG = re.compile(r"실적[^\n]{0,12}제한\s*(?:하지|하지는|은|을)?\s*(?:않|아니)|활동\s*실적|봉사\s*실적")


def evaluation_instruction_only(line: str) -> bool:
    """평가 안내에 쓰인 '숙지/유념하여야'를 실적 보유 의무로 읽지 않는다(§10-4 섹션인식).

    "평가항목·배점을 숙지하여야 합니다" 류 안내문은 실적 '보유' 요건이 아니라 응찰자 유의사항이다.
    같은 문장에 실제 실적 자격조건(보유/갖춘/N건 이상 등)이 함께 있으면 실적제한으로 그대로 남긴다.
    (후보킷 아이디어 이식 · dev200/20k 이중검증 대상)
    """
    if not (re.search(r"숙지|유념", line or "") and
            re.search(r"평가\s*(?:항목|기준|방법|배점)", line or "")):
        return False
    requirement = re.search(
        r"실적[^\n.;。]{0,50}(?:보유|갖춘|있는\s*(?:업체|자|법인|기업)|있어야|이상)"
        r"|(?:실적\s*제한|실적으로\s*제한)"
        r"|\d[^\n.;。]{0,25}(?:원|건)\s*이상[^\n.;。]{0,30}실적", line)
    return requirement is None


# [PATCH16] 적격심사 기준의 실적 '평가' 줄("이행실적의 당해용역 규모(금액기준): 금X원", "특별신인도는 … 실적합계액이 기초금액
#        이상인 경우 부여", "심사항목: 이행실적, 경영상태", "유사사업 수행실적 우대", "이행실적의 인정 여부는 발주부서 판단")은
#        참가자격 제한이 아니다(자가라벨 1,984건 v2 FP 12건 중 9건). 같은 줄에 자격 요구 표지(있는 업체·이어야·자로서·제한)가 있으면 유지.
_RE_PERF_EVAL_LINE = re.compile(r"특별\s*신인도|당해\s*용역\s*규모|규모\s*\(\s*금액\s*기준|심사\s*항목|평가\s*항목|이행실적\s*평가|"
                                r"인정\s*여부|우대|가점|배점")
_RE_PERF_STRONG_REQ = re.compile(r"있는\s*(?:업체|자|단체|법인|기업)|이어야|여야|자로서|업체로서|제한")


def _perf_line_ok(line: str) -> bool:
    if _RE_PERF_NEG.search(line):
        return False
    if evaluation_instruction_only(line):
        return False
    if _RE_PERF_EVAL_LINE.search(line) and not _RE_PERF_STRONG_REQ.search(line):
        return False
    if _RE_PERF_REQ.search(line):
        return True
    return not _RE_PERF_DOC.search(line)


def detect_perf_requirement(qual: str) -> Tuple[bool, str]:
    """_RE_PERF_DET 가 걸린 줄 중 실적을 '요구'하는 줄만 인정한다.

    [변경 전] 정규식 1회 검색 → "(정량평가) … 이행 실적증명서 각 1부 – 해당되는 경우만",
              "이행실적 평가는 … 실적합계액으로 평가합니다" 같은 평가·서류 줄도 실적제한으로 봄
              (비라벨 500건 v2 2.3x·v8 3.1x 과발화, dev FP 5건).
    [변경 후] 요구 표지(이상/보유/있는 업체/이어야/제한 …)가 없고 서류·평가 어휘만 있는 줄은 건너뜀.
    """
    for m in _RE_PERF_DET.finditer(qual or ""):
        line = _line_of(qual, m.start(), m.end())
        if _perf_line_ok(line):
            return True, line[:260]
    return False, ""


# v3 실적 1배수 초과 — 자격블록의 "실적" 줄에 적힌 요구 금액을 원 단위로 뽑는다.
# dev 200건 양성 8건이 모두 '3억원 이상 … 실적' 형태로 금액을 명시한다.
_RE_PERF_LINE = re.compile(r"[^\n]*실적[^\n]*")


# [추가] 적격심사 '평가기준' 줄은 참가자격 제한이 아니다. 자가라벨 2,112건 대조에서 v3 규칙 FP 20건 중 8건이
#        "이행실적의 당해용역 규모(금액기준): 금X원(추정금액)" · "특별신인도는 … 실적합계액이 기초금액 이상인 경우 부여"
#        같은 배점·가점 기준 줄이었다(dev 양성 8건은 모두 자격 요건 줄이라 영향 없음).
_RE_PERF_EVAL = re.compile(r"특별\s*신인도|신인도|평가\s*(?:기준|항목|점수|방법)|배점|만점|가점|"
                           r"규모\s*\(\s*금액\s*기준\s*\)|당해\s*용역\s*규모|(?:추정|기초)금액\)")


# [추가] v4 특정기관 실적 — 기관 어휘가 실적 '요구 조건' 줄에 있어야 한다. 자가라벨 20k 대조에서 규칙 FP 12건 중 8건이
#        "공공기관의 확인을 받은 실적증명서 제출"(증빙 안내) · "공공기관 이외의 실적은 계약서 등 첨부"(민간 실적도 인정) ·
#        "실적인정여부는 발주처([수요기관(공공기관)]) 의견에 따름" · 서류 제출 장소/담당자 연락처 · "용역명: 국가기관 산하기관 경영실적 평가"(제목)
#        줄이었다. 요구 표지(_perf_line_ok)가 없거나 제목·연락처·'이외의 실적' 줄이면 건너뛰고 다음 매치를 본다.
#        dev 양성 4건("국가기관·지자체에 납품한 실적이 있는 업체" 형태)은 모두 요구 표지가 있어 영향 없음(재생으로 확인).
_RE_PERF_ORG_SKIP = re.compile(r"^\s*[○◦•\-‣※]?\s*(?:용\s*역\s*명|사\s*업\s*명|공\s*고\s*명|건\s*명)\s*[:：]"
                               r"|☎|☏|전화번호|이외의\s*실적|인정\s*여부|의견에\s*따")


# "공공기관에서 발급한 실적증명서만 인정" · "공공기관에 대한 경영평가 실적이 있는 회계법인" 은 서류 어휘가 섞여 있어도 요구 조건이다.
_RE_PERF_ORG_REQ = re.compile(r"만\s*인정|실적이\s*있는|실적을\s*보유")


def detect_perf_org(qual: str) -> Tuple[bool, str]:
    for m in _RE_PERF_ORG.finditer(qual or ""):
        line = _line_of(qual, m.start(), m.end())
        if _RE_PERF_ORG_SKIP.search(line):
            continue
        if not (_RE_PERF_ORG_REQ.search(line) or _perf_line_ok(line)):
            continue
        return True, line.strip()[:300]
    return False, ""


def detect_perf_amount(qual: str) -> Tuple[int, str]:
    """참가자격에서 요구하는 실적 금액의 최대값과 그 줄을 돌려준다."""
    best, quote = 0, ""
    for m in _RE_PERF_LINE.finditer(qual or ""):
        line = m.group(0)
        if _RE_PERF_EVAL.search(line):
            continue
        amt = parse_amount_won(line)
        if amt > best:
            best, quote = amt, line.strip()[:300]
    return best, quote

_PUB_ORG = r"국가기관|공공기관|지방자치단체|지자체|정부기관|정부투자기관|관공서"
_RE_PERF_ORG = re.compile(
    rf"실적[^\n]{{0,50}}(?:{_PUB_ORG})|(?:{_PUB_ORG})[^\n]{{0,50}}실적"
    rf"|(?:기관|청|부|공사|공단|대학|병원|지방자치단체)\s*에\s*납품(?:한|된)")

# v9 특정 모델명 명시 — dev 200건 TP2/FP0 (P 1.000 · R 0.333)
# 규격서 특정 모델 지정(v9). 세 번째 대안은 "Matrice 4E/T 시리즈"처럼
# 라틴문자 상표명 + 모델번호 + '시리즈' 형태의 제품 계열 지정을 잡는다.
# dev 기준 FP 0 을 유지하면서 재현율 0.333 → 0.500 (F1 0.500 → 0.667).
# [추가] 규격서 표의 '모델명' 열 아래 셀에 "Agilent ICP-OES 5900" 처럼 라틴 상표 + 숫자 포함 모델코드가 적힌 경우(dev 052).
#        문장 속 "모델명을 명시하여야" 류(dev 082·095·152·158·180, 정답 0)는 표 형태가 아니라 걸리지 않는다.
#        무라벨 20k 발화 7건(0.035%) 모두 실제 모델 지정(Dell PowerEdge R660xs·Kubota V1505 등).
# [변경] "모델명 :" 뒤가 비어 있는 서식(입찰자 기재란·규격확인서·작성요령 "계약장비의 모델명 기록")은 지정이 아니다
#        (무라벨 20k v9 FP 14건). 콜론 뒤 40자 안에 영숫자 모델값이 있고 기록·기재·작성 안내가 아닐 때만 인정한다.
_RE_MODEL_DET = re.compile(r"(?<!응찰\s)모델\s*(?:명|번호)\s*[:：] ?(?!\s)(?![^\n]{0,15}(?:기록|기재|작성))(?=[^\n]{0,40}[A-Za-z0-9])[^\s\n]"
                           # [변경] "규격서에 기재된 성능(사양)과 동등 이상"은 성능 기준 서술이지 모델 지정이 아니다(자가라벨 v9 FP
                           #        PPS-D-018038 초분광 카메라 성능표). "기재된 내용과 동등 이상"(식약처 허가제품 문구)은 dev 051 정답 1 → 유지.
                           r"|규격서에\s*기재된\s*(?!\s*(?:성능|사양))[^\n]{0,20}동등\s*이상"
                           r"|\b[A-Z][A-Za-z]{1,}\s+\d{1,2}[A-Z]?(?:/[A-Z])?\s*시리즈"
                           r"|(?:^|\n)\s*모델\s*명\s*\n(?:[^\n]{0,60}\n){0,14}?\s*"
                           r"[A-Z][a-z]{2,}(?:[ -][A-Za-z]{2,}){0,3}\s+[A-Z0-9][A-Za-z0-9\-/]*\d[A-Za-z0-9\-/]*\s*\n"
                           # [PATCH13b] 자가라벨 v9 FN: "대표규격 MIR-554", "제조사(모델명): TECORA(DECS)", "제작사 LUTRON 모델명 GRAFIK Eye QS",
                           #        "모델명 GSL-O50"(콜론 없이 라틴 모델값이 바로 이어짐), "압축기 형식: DURR VS 900 HEAD".
                           r"|대표\s*규격\s*[:：]?\s*[A-Za-z][A-Za-z0-9\-]*\d[A-Za-z0-9\-()]*"
                           r"|제[조작]사\s*\(\s*모델\s*명\s*\)\s*[:：]\s*[A-Za-z]"
                           r"|모델\s*명[ \t]+(?![^\n]{0,15}(?:기록|기재|작성|란|양식))[A-Z][A-Za-z0-9\-]{2,}(?:\s+[A-Za-z0-9\-]{1,}){0,2}"
                           r"|형식\s*[:：]\s*[A-Z]{3,}(?:[ -][A-Za-z0-9]{2,}){1,3}"
                           # [PATCH16] 자가라벨 v9 FN(1,984건 대조): 표의 파이프 셀 "모델명 | JV-45RD", 제조사를 괄호로 지정
                           #   "제조사【삼성전자】/〔한국타피(주)〕", "입찰대상물품(특정제품)", "브랜드(파나소닉, NEC, 앱손) 제품에 한해",
                           #   "삼성,엘지에서 제조한 정품". 무라벨 1,984건 발화 7건 모두 자가라벨 v9=1(high), dev200 발화 0.
                           r"|모델\s*명\s*\|\s*[A-Za-z0-9\-]*\d[A-Za-z0-9\-]*\s*(?:\||\n|$)"
                           r"|제조사\s*[〔\[【]\s*[^\n〕\]】]{2,20}[〕\]】]"
                           r"|입찰대상물품\s*\(\s*특정제품\s*\)"
                           r"|브랜드\s*\([^)\n]{2,40}\)\s*제품에\s*한"
                           r"|(?:삼성|엘지|LG|델|Dell|HP|레노버|애플|Apple)[,·/ ]*(?:삼성|엘지|LG|델|HP|레노버)?\s*에서\s*제조한")

# 기업규모 제한 — "중소기업" 안에 "소기업"이 들어있어 단순 검색으로는 분리되지 않는다.
# 실제 공고는 근거법 인용과 확인서 이름으로 둘을 구분한다:
#   중소기업 제한 : 「중소기업기본법」…에 따른 중소기업 / <중소기업·소상공인 확인서>
#   소기업 제한   : 「중소기업기본법」…에 따른 소기업   / <소기업·소상공인 확인서>
# [변경] 구분자에 '/' 추가("소기업/소상공인확인서" 표기가 비라벨 데이터에 다수) ·
#        "판로지원법 시행령 제2조의2 에 의거 중소기업확인서" 형태(확인서만 명시)를 중소기업 제한으로 인정.
_SEP = r"[·ㆍ・․‧･∙•,\.\s/]?"   # [변경] U+2024·U+2027 등 비표준 가운뎃점 포함(무라벨 20k v11 FP: "중‧소기업‧소상공인 확인서")
# [추가] "중소기업 또는 소상공인으로서 … 소기업·소상공인 확인서를 소지한 업체"(dev 076·077) 처럼 규모 정의는
#        중소기업으로 인용하되 실제 요구 확인서가 '소기업·소상공인 확인서'면 소기업·소상공인 제한이다.
_SEP_W = r"\s*[·ㆍ・․‧･∙•,\./]?\s*"
_RE_SMALL_CERT_REQ = re.compile(
    rf"(?<!중)소기업{_SEP_W}(?:또는|및)?\s*소상공인\s*확인서[^\n]{{0,40}}?(?:소지|보유|제출|발급)"
    )
# [PATCH16] "소기업 또는 소상공인으로 자격을 제한"·"(소기업,소상공인으로 제한)" 처럼 제한 선언을 직접 적은 문구는
#        근거법 인용("중소기업기본법에 따른 중소기업자로서")·확인서 명칭보다 우선한다(자가라벨 v17 FP 001270·002681·019773).
_RE_SMALL_EXPLICIT = re.compile(
    r"(?<!중)소기업\s*(?:또는|및|[,·ㆍ])\s*[「\"“]?[^\n]{0,40}?소상공인(?:으로|만|에)\s*(?:입찰\s*)?(?:참가\s*)?(?:자격을\s*)?(?:제한|한정)(?!하는|되는|할\s*경우|한\s*경우|시)")   # 적격심사기준의 조건절("…으로 제한하는 입찰에서는") 제외
_RE_SME_CERT_REQ = re.compile(
    rf"중소기업{_SEP_W}(?:또는|및)?\s*소상공인\s*확인서[^\n]{{0,40}}?(?:소지|보유|제출|발급)")
_RE_SIZE_SME = re.compile(
    rf"중소기업기본법[^\n]{{0,40}}따른\s*중{_SEP}\s*소기업"
    rf"|중{_SEP}소기업{_SEP}\s*(?:또는|및)?\s*소상공인\s*확인서"
    rf"|제2조의\s*2[^\n]{{0,20}}중소기업\s*확인서"
    # [추가] "중소기업확인서(입찰 마감일 전일까지 발급된 것)를 소지한 자"·"중소기업자간경쟁입찰"(무라벨 20k v11 FP 6건)
    rf"|중소기업\s*확인서(?!\s*\(\s*소기업)[^\n]{{0,60}}?(?:소지|보유|제출|발급)"
    rf"|중소기업자\s*간\s*경쟁\s*입찰(?![^\n]{{0,12}}예외)"
    rf"|(?:따른|의한)\s*중소기업\s*(?:또는|및|,)\s*[「\"]?\s*소상공인"
    rf"|요건을\s*갖춘\s*중{_SEP}소기업자"
    rf"|따른\s*중기업\s*(?:또는|이거나|및)\s*소기업")
_RE_SIZE_SMALL = re.compile(
    rf"중소기업기본법[^\n]{{0,40}}(?:따른|의한)\s*소기업"
    rf"|(?<!중)소기업{_SEP}\s*(?:또는|및)?\s*소상공인\s*확인서"
    # [변경] "소기업 또는 「소상공인 보호 및 지원에 관한 법률」에 의한 소상공인"(여는 낫표)·"중소기업확인서(소기업, 소상공인)"·
    #        "소상공인․소기업 확인서"(U+2024 구분자) 를 놓쳐 중소기업 제한으로 오분류(무라벨 20k v17 FP·v18 FN).
    rf"|(?<!중)소기업\s*(?:또는|및|,|·|ㆍ)\s*[「\"“]?\s*소상공인"
    rf"|중소기업\s*확인서\s*\(\s*소기업\s*[,·ㆍ]?\s*소상공인\s*\)"
    rf"|(?<!중)소기업\s*\(\s*소상공인\s*\)"
    rf"|특별조치법[^\n]{{0,30}}따른\s*소기업"
    # [추가] 무라벨 20k v18 FP 분석: 따옴표 감싼 "소기업" 또는 …"소상공인", 순서가 바뀐 "소상공인 또는 소기업 확인서",
    #        "소기업과 「소상공인 보호…」" 형태를 놓쳐 소기업 제한이 있는데도 부재탐지(v18)가 발화했다.
    rf"|따른\s*[\"“]소기업[\"”]\s*(?:또는|및|,)"
    rf"|소상공인{_SEP_W}(?:또는|및)?\s*(?<!중)소기업\s*(?:확인서|에\s*해당|이어야|여야|으로|만)"
    rf"|(?<!중)소기업과\s*[「\"“]?\s*소상공인")

# v12 일반제품 직접생산확인 제한 — dev 200건 TP3/FP0 (P 1.000)
# [변경] 증명서 뒤에 "[세부품명 : ○○○(10자리 코드)]" 괄호 설명이 길게 끼어 30자 창을 넘던 공고(dev 053·054·056·169)를
#        위해 "직접생산확인증명서 … 를 소지(보유·제출)한 업체" 형태를 두 번째 패턴으로 추가한다.
_RE_DPC_DET = re.compile(r"직접\s*생산\s*확인[^\n]{0,30}(?:업체|자)(?:이어야|여야|만)"
                         r"|직접\s*생산\s*확인\s*증명서[^\n]{0,120}?(?:소지|보유|제출)한?\s*(?:업체|자)")
# 부재탐지(v10)는 "언급조차 없음"을 봐야 하므로 넓게 잡는다. 제한판정과 스코프가 다르다.
# [변경] "직접생산증명서"(판로지원법 제9조·시행령 제10조 표기)도 직생 요구다 — dev 060·074·138 v10 과탐 원인.
# [변경] 무라벨 20k v10 FP 21건 중 15건이 "직접생 산확인증명서"(줄바꿈 공백)·"직적생산확인증명서"(오타)·
#        "직접 생산 확인 증명서"·자격블록 밖(제출서류 목록·규격서)에 적힌 문구였다 → 글자 사이 공백·오타 허용, 전문에서 찍는다.
_RE_DPC_ANY = re.compile(r"직\s*[접적]\s*생\s*산\s*(?:확\s*인|증\s*명)")
# 전문 탐색은 "요구 문장"(…증명서를 소지한 업체/이어야) 에 한정한다 — 제출서류 목록의 "직접생산확인증명서 1부"·
# "직접생산 확인기준을 위반한 경우 계약해지" 보일러플레이트는 dev(013·064·069·075) 에서 v10 위반으로 판정된다.
# [변경] "직접생산자확인증명서"(자 삽입)·괄호 설명 중간의 단일 줄바꿈(빈 줄 제외)을 허용한다 — 무라벨 20k v10 FP(자가라벨 대조) 2건.
_RE_DPC_REQ_FULL = re.compile(r"직\s*[접적]\s*생\s*산\s*자?\s*(?:확\s*인)?\s*증\s*명\s*서(?:[^\n]|\n(?!\s*\n)){0,160}?(?:소지|보유|이어야|여야|에\s*한(?:함|하))")

# 판로지원법 시행령 제2조의3 — 중소기업자 우선조달계약의 예외.
# 이 문구가 있으면 규모 제한이 "없어도" 적법하므로 부재탐지(v16·v18)를 끈다.
# [변경] "제2조의3" 단독 매칭은 「인지세법 시행령」 제2조의3(인지세 납부) 을 예외조항으로 오인했다(무라벨 20k v11 FP).
#        판로지원법·우선조달 문맥이 앞뒤 40자 안에 있을 때만 인정하고, 운영요령·업무처리기준의 예외 적용 문구를 추가한다.
_RE_EXC_DET = re.compile(
    r"우선조달계약에?\s*대한\s*예외|예외가\s*적용되는\s*사업"
    r"|(?:판로지원|우선조달|공공구매)[^\n]{0,40}제2조의\s*3|제2조의\s*3[^\n]{0,30}(?:우선조달|예외)"
    r"|판로지원[^\n]{0,40}예외|운영요령[^\n]{0,40}예외|업무처리기준[^\n]{0,40}예외")


_RE_SIZE_NEG_LINE = re.compile(r"제한\s*(?:이|은|을)?\s*(?:없습니다|없음|없다|없으며|두지\s*않|하지\s*않)|비\s*대상")


def detect_size_limit(qual: str) -> Tuple[str, str]:
    """참가자격 블록에서 기업규모 제한을 결정적으로 분류한다."""
    if not qual:
        return "", ""
    # [변경] "중․소기업자"·"중 · 소기업자"(U+2024 등 비표준 구분자, 공백 혼입)는 '중소기업'이다.
    #        예전엔 (?<!중)소기업 이 걸려 소기업 제한으로 오분류 → v15 FP(dev 121·128). 판정용 사본만 정규화하고
    #        인용문은 원문에서 뽑는다.
    # [PATCH13c] "고시금액 이상 입찰로 소기업·소상공인·중소기업자간 제한 입찰 등의 제한이 없습니다" ·
    #        "중소기업자간 제한경쟁입찰 비대상"처럼 규모 제한을 '부정'하는 줄은 판정에서 제외한다(자가라벨 v14 FP 000104·009886).
    qual = "\n".join(ln for ln in qual.split("\n") if not _RE_SIZE_NEG_LINE.search(ln))
    qn = re.sub(r"중\s*[·ㆍ・․‧･∙•,\./]?\s*소기업", "중소기업", qual)
    # "중기업·소기업 또는 소상공인"처럼 중기업을 함께 허용하면 중소기업 전체 제한이다(dev 117·169 v13 FP).
    # [변경] 구분자 중복("중기업·․소기업")·슬래시("중기업/소기업/소상공인")도 같은 뜻이다(무라벨 20k v13 FP 5건, dev 6건 모두 정답 0).
    qn = re.sub(r"중기업자?\s*[·ㆍ・․‧･∙•,\./]*\s*(?:및\s*|또는\s*)?소기업", "중소기업", qn)
    # [추가] 자가라벨 20k v15 FP 15건: 중기업을 함께 허용하는 표기 변형들은 모두 중소기업 전체 제한이다.
    #   "중소기업(소기업 또는 소상공인 포함)" · "중소기업ㆍ소기업ㆍ소상공인 확인서" · "소기업 및 중기업 또는 소상공인" ·
    #   "중기업확인서, 소기업·소상공인확인서"(중기업 확인서도 인정) — 자격블록에 '중기업'이 독립 등장하면 중기업 배제가 아니다.
    qn = re.sub(r"중소기업\s*\(\s*소기업\s*(?:또는|및|[·ㆍ・,])\s*소상공인\s*포함\s*\)", "중소기업", qn)
    qn = re.sub(r"중소기업\s*[·ㆍ・․‧･∙•,\./]\s*소기업\s*[·ㆍ・․‧･∙•,\./]\s*소상공인", "중소기업소상공인", qn)
    qn = re.sub(r"소기업\s*(?:및|또는|[·ㆍ・,/])\s*중기업", "중소기업", qn)
    # [PATCH16] "중기업 및 대기업은 참여할 수 없습니다"는 중기업 '배제' 선언이다 — 중기업 허용으로 읽지 않는다(자가라벨 v17 FP 001759).
    mid_ok = bool(re.search(r"중기업", re.sub(r"중기업[^\n]{0,25}(?:참여|참가|입찰)[^\n]{0,12}(?:없|불가|제외|배제|못)", "", qn)))
    # 법령명 "소기업 및 소상공인 지원을 위한 특별조치법"과 실적 인정기간 완화 문구 "(창업기업, 소기업 및 소상공인은 7년)"은
    # 규모 제한 선언이 아니다(무라벨 20k v13 FP).
    qn = re.sub(r"소기업\s*및\s*소상공인\s*지원을\s*위한", "특별조치법명", qn)
    qn = re.sub(r"소기업\s*(?:및|,)\s*소상공인은\s*\d+\s*년", "실적기간완화", qn)
    sme, small = _RE_SIZE_SME.search(qn), _RE_SIZE_SMALL.search(qn)
    sc, mc = _RE_SMALL_CERT_REQ.search(qn), _RE_SME_CERT_REQ.search(qn)
    if not mid_ok and _RE_SMALL_EXPLICIT.search(qn):
        return "소기업소상공인", _first_line(qual, _RE_SMALL_EXPLICIT) or _first_line(qual, _RE_SIZE_ANY)
    if sme and sc and (mc is None or sc.start() < mc.start()):
        return "소기업소상공인", _first_line(qual, _RE_SMALL_CERT_REQ) or _first_line(qual, _RE_SIZE_ANY)
    if small and mid_ok:
        return "중소기업", _first_line(qual, _RE_SIZE_SME) or _first_line(qual, re.compile("중기업")) or _first_line(qual, _RE_SIZE_ANY)
    if small and not sme:
        return "소기업소상공인", _first_line(qual, _RE_SIZE_SMALL) or _first_line(qual, _RE_SIZE_ANY)
    if small and sme and small.start() < sme.start():
        # 두 규모가 함께 등장하면 자격블록에서 '먼저' 선언된 쪽이 실제 제한이다.
        # (뒤쪽은 대개 근거법령·정의 인용) — 변형 7종 중 dev 최고(0.7237).
        return "소기업소상공인", _first_line(qual, _RE_SIZE_SMALL) or _first_line(qual, _RE_SIZE_ANY)
    if sme:
        return "중소기업", _first_line(qual, _RE_SIZE_SME) or _first_line(qual, _RE_SIZE_ANY)
    return "", ""


_RE_SIZE_ANY = re.compile(r"소기업|소상공인|중소기업자\s*간\s*경쟁")
_RE_SIZE_REQ_LINE = re.compile(r"이어야|여야\s*(?:합니다|한다|함)|우선조달계약\s*대상|참가\s*(?:가능|할\s*수)|에\s*한(?:함|하)|확인서를?\s*(?:소지|보유|제출)|중소기업자\s*간\s*경쟁\s*입찰(?![^\n]{0,12}예외)")


def _first_line(text: str, rx: "re.Pattern") -> str:
    """패턴이 걸린 지점을 포함하는 한 줄을 근거 인용문으로 돌려준다."""
    m = rx.search(text)
    if not m:
        return ""
    a = text.rfind("\n", 0, m.start()) + 1
    b = text.find("\n", m.end())
    return text[a: b if b > 0 else len(text)].strip()[:300]


def backstop_facts(qual: str, full: str) -> Dict[str, Any]:
    """자격블록·전문에서 LLM 전용 항목의 결정론적 신호를 뽑는다."""
    out: Dict[str, Any] = {}
    if qual:
        # [변경] 자가라벨 대조(무라벨 20k 층화표본)에서 v1 규칙 양성의 절반이 상투 문구였다:
        #   "…물품 수량을 확보하여 발주기관에 납품이 가능한 경우에만 입찰에 참가" (조건절 '경우에만' + 발주기관)
        #   "중소기업협동조합은 … 적격조합이어야 하며 … 제출할 경우에만 입찰참여 가능" (경쟁제품 공고 표준 문구)
        # → 해당 줄은 기관 유형 한정이 아니므로 제외한다.
        inst_q = ""
        for m in _RE_INST.finditer(qual):
            a = qual.rfind("\n", 0, m.start()) + 1
            b = qual.find("\n", m.end())
            line = qual[a: b if b > 0 else len(qual)]
            # [변경] 자가라벨 20k: "○○협동조합으로부터 추천 받은 N개 업체만 입찰 가능"은
            #   진짜 참가자격 제한(v1)인데 _RE_INST_BOILER의 '협동조합' 항목에 걸려 억제됐다(FN 4건).
            #   '추천 받은 … 업체만' 강한 제한 신호가 있으면 보일러플레이트 예외로 되살린다(신규 FP 0건 확인).
            if _RE_INST_BOILER.search(line) and not _RE_INST_RECOMMEND.search(line):
                continue
            inst_q = line.strip()[:300]
            break
        out["inst_det"] = bool(inst_q)
        out["inst_quote"] = inst_q
        out["perf_org_det"], out["perf_org_quote"] = detect_perf_org(qual)
        # [추가] "국가, 지방자치단체, 공공기관, 민간(기업) 실적" 처럼 민간을 함께 인정하면 공공기관 한정이 아니다(무라벨 20k v4 FP).
        if out["perf_org_det"] and re.search(r"민간", out["perf_org_quote"] or ""):
            out["perf_org_det"], out["perf_org_quote"] = False, ""
        # [변경] 실적 줄이 '요구 조건'인지 검사(정량평가 항목·제출서류 목록 제외) → v2/v8 과탐 억제
        # [대폭 수정 2026-09-19b] detect_perf_requirement는 _RE_PERF_DET 패턴만 사용하여
        # "유사용역 수행실적" 같은 일반적 실적 요구를 놓침. perf_requirement_line()은
        # _RE_PERF_WORD(더 넓은 패턴) + _perf_line_ok()를 써서 재현율이 높다.
        _prl = perf_requirement_line(qual)
        if _prl:
            out["perf_det"], out["perf_quote"] = True, _prl
        else:
            out["perf_det"], out["perf_quote"] = detect_perf_requirement(qual)
        out["perf_amt_det"], out["perf_amt_q"] = detect_perf_amount(qual)
    else:
        # [대폭 수정 2026-09-19b] qual이 비어도 전문(full)에서 실적/기관 패턴 탐색
        _prl_full = perf_requirement_line(full[:8000]) if full else ""
        out["perf_det"] = bool(_prl_full)
        out["perf_quote"] = _prl_full
        out["perf_amt_det"], out["perf_amt_q"] = detect_perf_amount(full[:8000]) if full else (0, "")
        out["inst_det"] = False
        out["inst_quote"] = ""
        out["perf_org_det"], out["perf_org_quote"] = False, ""
        if full:
            # 전문에서 기관 제한 탐지 (qual이 없는 경우)
            _inst_q_full = ""
            for _m in _RE_INST.finditer(full[:6000]):
                _a = full.rfind("\n", 0, _m.start()) + 1
                _b = full.find("\n", _m.end())
                _line = full[_a: _b if _b > 0 else len(full)]
                if _RE_INST_BOILER.search(_line) and not _RE_INST_RECOMMEND.search(_line):
                    continue
                _inst_q_full = _line.strip()[:300]
                break
            out["inst_det"] = bool(_inst_q_full)
            out["inst_quote"] = _inst_q_full
            out["perf_org_det"], out["perf_org_quote"] = detect_perf_org(full[:6000])
            if out["perf_org_det"] and re.search(r"민간", out["perf_org_quote"] or ""):
                out["perf_org_det"], out["perf_org_quote"] = False, ""
    out["model_det"] = bool(_RE_MODEL_DET.search(full))
    out["model_quote"] = _first_line(full, _RE_MODEL_DET)
    out["size_det"], out["size_det_q"] = detect_size_limit(qual)
    if out["size_det"] == "중소기업" and full and len(qual) >= 3500:
        # [추가] 자격블록이 3,600자 한계에서 잘려 '소기업·소상공인 확인서를 소지한 자' 요구 줄이 빠지고
        #        뒤쪽 안내문("중소기업ㆍ소상공인확인서를 … 신청")만 담긴 공고(자가라벨 v17 FP PPS-D-020735).
        #        전문에서 소기업·소상공인 확인서 '요구'가 있고 전문 판정도 소기업·소상공인이면 그쪽을 따른다.
        if _RE_SMALL_CERT_REQ.search(full):
            det_full, q_full = detect_size_limit(full)
            if det_full == "소기업소상공인":
                out["size_det"], out["size_det_q"] = det_full, q_full
    if not out["size_det"] and full and len(qual) >= 3500:
        # [추가] 자격블록이 3,600자 한계에서 잘려 뒤쪽 규모 제한 조항("중‧소기업‧소상공인 확인서를 소지한 자")을
        #        놓친 공고(무라벨 20k v11 FP: PPS-D-007488·008547). 판정용으로만 더 긴 블록을 다시 뽑는다.
        out["size_det"], out["size_det_q"] = detect_size_limit(extract_qualification(full, 9000))
    if not out["size_det"] and full:
        # [추가] 자격블록(최대 3,600자)이 잘려 규모 제한 조항이 빠진 공고(무라벨 20k v18 FP: PPS-D-000579 등).
        #        전문에서 '요구 조건' 성격의 줄(이어야·대상 입찰·확인서 소지 등)에 한해 다시 찍는다.
        for line in full.split("\n"):
            if not _RE_SIZE_ANY.search(line) or not _RE_SIZE_REQ_LINE.search(line):
                continue
            det, q = detect_size_limit(line)
            if det:
                out["size_det"], out["size_det_q"] = det, q
                break
    out["exc_det"] = bool(_RE_EXC_DET.search(full))
    out["dpc_det"] = bool(_RE_DPC_DET.search(qual)) if qual else False
    out["dpc_quote"] = _first_line(qual, _RE_DPC_DET) if qual else ""
    out["dpc_any"] = bool(_RE_DPC_ANY.search(qual)) if qual else False
    out["dpc_any"] = out["dpc_any"] or bool(_RE_DPC_REQ_FULL.search(full))
    return out


# ===================================================================================
# 8. 모델 러너
# ===================================================================================
def v_token_lnodds(comp: Any) -> Optional[float]:
    """판정 JSON 의 "v": 값 위치에서 부호 있는 ln-odds  logP(1) - logP(0) 을 구한다(없으면 None).

    [추가] G) conformal abstain 및 [2026-09-20] logprob 임계 튜닝용. JSON 스키마가 v 를 enum[0,1] 로
    강제하므로 그 자리는 사실상 이진 결정이고, 두 후보의 logprob 차이가 곧 판정의 확신도다.
    · 구조화 출력에서는 '1,' 처럼 숫자 뒤에 구분자가 붙은 단일 토큰이 나올 수 있어 "0/1 로 시작하는
      토큰"까지 인식한다.
    · 대안 숫자가 top-k 밖이면 매우 확신한 것으로 보고 ±9.0 을 돌려준다.
    """
    lps = getattr(comp, "logprobs", None)
    ids = getattr(comp, "token_ids", None)
    if not lps or not ids:
        return None
    acc = ""
    for tid, d in zip(ids, lps):
        if not isinstance(d, dict):
            return None
        ent = d.get(tid)
        tok = (getattr(ent, "decoded_token", None) or "") if ent is not None else ""
        prev = acc
        acc += tok
        t0 = tok.strip()
        if t0[:1] in ("0", "1") and prev.replace(" ", "").rstrip().endswith('"v":'):
            chosen = t0[0]
            lp: Dict[str, float] = {}
            for e in d.values():
                t = (getattr(e, "decoded_token", "") or "").strip()[:1]
                if t in ("0", "1"):
                    v = getattr(e, "logprob", None)
                    if v is not None and (t not in lp or float(v) > lp[t]):
                        lp[t] = float(v)
            if lp.get("0") is None or lp.get("1") is None:
                return 9.0 if chosen == "1" else -9.0          # 대안이 top-k 밖 → 사실상 확정
            return float(lp["1"]) - float(lp["0"])
        if len(acc) > 4000:                               # 방어: 비정상적으로 긴 출력
            break
    return None


def v_token_margin(comp: Any) -> Optional[float]:
    """|logP(1) - logP(0)| (기권 판정용, 하위호환)."""
    lo = v_token_lnodds(comp)
    return None if lo is None else abs(lo)


class VLLMRunner:
    """평가 서버 고정 모델을 vLLM offline API로 실행."""

    def __init__(self, schema: Dict[str, Any], model_dir: str = MODEL_DIR,
                 quant: Optional[str] = QUANT, max_tokens: int = MAX_TOKENS,
                 seed: int = SEED, gpu_mem: float = 0.92, tp: int = 1,
                 max_model_len: int = MAX_MODEL_LEN):
        t0 = time.time()
        import vllm
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        log(f"vllm {vllm.__version__} · {model_dir} · quant={quant} · max_model_len={max_model_len}")
        kw: Dict[str, Any] = dict(model=model_dir, tokenizer=model_dir, max_model_len=max_model_len,
                                  gpu_memory_utilization=gpu_mem, seed=seed,
                                  tensor_parallel_size=tp, dtype="auto")
        if quant:
            kw["quantization"] = quant
        self.llm = LLM(**kw)
        self.tok = self.llm.get_tokenizer()
        self.max_model_len = max_model_len
        self.max_tokens = max_tokens
        self.sp = SamplingParams(
            temperature=0.0, max_tokens=max_tokens, seed=seed,
            structured_outputs=StructuredOutputsParams(json=schema, disable_any_whitespace=True),
        )
        self.load_seconds = time.time() - t0

    def count_tokens(self, messages: List[Dict[str, str]]) -> int:
        try:
            ids = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
            if hasattr(ids, "keys") and "input_ids" in ids:
                ids = ids["input_ids"]
            return len(ids)
        except Exception:
            return len(self.tok.encode("\n".join(m["content"] for m in messages)))

    def chat(self, batch: List[List[Dict[str, str]]]) -> List[str]:
        outs = self.llm.chat(batch, sampling_params=self.sp, use_tqdm=False)
        return [o.outputs[0].text if o.outputs else "" for o in outs]

    def set_judge(self, schema: Dict[str, Any], max_tokens: int) -> None:
        """2차 판정용 샘플링 파라미터(별도 JSON 스키마·짧은 출력).

        [변경] JUDGE_SAMPLES>1 이면 온도 샘플링 n회(자기일관성 다수결용), 1이면 기존 그리디.
               JUDGE_LOGPROBS>0 이면 v 토큰의 logprob 을 함께 받아 기권 판정에 쓴다.
        """
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
        kw: Dict[str, Any] = dict(
            max_tokens=max_tokens, seed=SEED,
            structured_outputs=StructuredOutputsParams(json=schema, disable_any_whitespace=True),
        )
        if JUDGE_SAMPLES > 1:
            kw.update(n=JUDGE_SAMPLES, temperature=JUDGE_TEMP, top_p=JUDGE_TOP_P)
        else:
            kw.update(temperature=0.0)
        if JUDGE_LOGPROBS > 0:
            kw["logprobs"] = JUDGE_LOGPROBS
        try:
            self.sp_judge = SamplingParams(**kw)
        except Exception as e:                            # 구버전 vLLM: n/logprobs 조합 미지원
            log(f"  ! 판정 샘플링 설정 실패({type(e).__name__}: {e}) → 그리디 단일 샘플로 대체")
            self.sp_judge = SamplingParams(
                temperature=0.0, max_tokens=max_tokens, seed=SEED,
                structured_outputs=StructuredOutputsParams(json=schema, disable_any_whitespace=True),
            )

    def chat_judge(self, batch: List[List[Dict[str, str]]]) -> List[str]:
        outs = self.llm.chat(batch, sampling_params=self.sp_judge, use_tqdm=False)
        return [o.outputs[0].text if o.outputs else "" for o in outs]

    def chat_judge_n(self, batch: List[List[Dict[str, str]]]) -> List[List[Dict[str, Any]]]:
        """프롬프트당 전체 샘플을 [{text, margin}] 목록으로 돌려준다(다수결·기권용)."""
        outs = self.llm.chat(batch, sampling_params=self.sp_judge, use_tqdm=False)
        res: List[List[Dict[str, Any]]] = []
        for o in outs:
            cand = []
            for c in (o.outputs or []):
                lo = v_token_lnodds(c)
                cand.append({"text": c.text, "margin": None if lo is None else abs(lo), "lo": lo})
            res.append(cand or [{"text": "", "margin": None}])
        return res


class MockRunner:
    """모델 없이 입·출력 흐름과 제출 형식을 확인한다."""
    load_seconds = 0.0

    def __init__(self, schema: Dict[str, Any], **_):
        self.schema = schema
        self.max_model_len = MAX_MODEL_LEN
        self.max_tokens = MAX_TOKENS

    def count_tokens(self, messages: List[Dict[str, str]]) -> int:
        return sum(len(m["content"]) for m in messages) // 2

    def chat(self, batch: List[List[Dict[str, str]]]) -> List[str]:
        base = {
            "inst": 0, "inst_q": None, "perf": 0, "perf_amt": 0, "perf_org": 0, "perf_q": None,
            "reg": 0, "reg_lv": "없음", "reg_q": None, "size": "없음", "size_q": None,
            "dpc": 0, "dpc_q": None, "cmp": 0, "model": 0, "model_q": None,
            "pled": 0, "pled_q": None, "brief": 0, "brief_q": None,
            "sw": 0, "swlim": 0, "exc": 0, "share": -1, "share_q": None,
            "mism": 0, "mism_q": None, "brief_date": None, "prop_due": None,
        }
        return [json.dumps(base, ensure_ascii=False) for _ in batch]

    def set_judge(self, schema: Dict[str, Any], max_tokens: int) -> None:
        pass

    def chat_judge(self, batch: List[List[Dict[str, str]]]) -> List[str]:
        # 모의 판정: 저신뢰로 판정을 보류(=규칙 결과 유지)
        return [json.dumps({"why": "mock", "v": 0, "conf": "low", "q": None}, ensure_ascii=False)
                for _ in batch]

    def chat_judge_n(self, batch: List[List[Dict[str, str]]]) -> List[List[Dict[str, Any]]]:
        return [[{"text": t, "margin": None}] for t in self.chat_judge(batch)]


def truncate_messages(messages: List[Dict[str, str]], runner, reserve: int) -> List[Dict[str, str]]:
    """토큰 예산을 넘으면 사용자 프롬프트 뒤쪽을 잘라 맞춘다."""
    budget = getattr(runner, "max_model_len", MAX_MODEL_LEN) - reserve
    n = runner.count_tokens(messages)
    if n <= budget:
        return messages
    body = messages[1]["content"]
    for ratio in (0.8, 0.62, 0.48, 0.36, 0.26, 0.18):
        cut = int(len(body) * ratio)
        trial = [messages[0], {"role": "user", "content": body[:cut] + "\n…(길이 예산으로 절단)"}]
        if runner.count_tokens(trial) <= budget:
            return trial
    return [messages[0], {"role": "user", "content": body[:2000]}]


def safe_chat(runner, batch: List[List[Dict[str, str]]], judge: bool = False) -> List[str]:
    """배치 실패 시 1건씩 재시도해 전체 중단을 막는다."""
    fn = runner.chat_judge if judge else runner.chat
    try:
        return fn(batch)
    except Exception as e:
        log(f"배치 추론 실패({type(e).__name__}: {e}) → 개별 재시도")
        out = []
        for m in batch:
            try:
                out.append(fn([m])[0])
            except Exception as e2:
                log(f"  개별 실패: {type(e2).__name__}: {e2}")
                out.append("")
        return out


def safe_chat_judge_n(runner, batch: List[List[Dict[str, str]]]) -> List[List[Dict[str, Any]]]:
    """다중 샘플 판정. 러너가 chat_judge_n 을 지원하지 않으면 단일 샘플로 감싼다."""
    fn = getattr(runner, "chat_judge_n", None)
    if fn is None:
        return [[{"text": t, "margin": None}] for t in safe_chat(runner, batch, judge=True)]
    try:
        return fn(batch)
    except Exception as e:
        log(f"다중 샘플 판정 실패({type(e).__name__}: {e}) → 개별 재시도")
        out: List[List[Dict[str, Any]]] = []
        for m in batch:
            try:
                out.append(fn([m])[0])
            except Exception as e2:
                log(f"  개별 실패: {type(e2).__name__}: {e2}")
                out.append([{"text": "", "margin": None}])
        return out


# ===================================================================================
# 8-1. 법령 조문 색인 + bge-m3 임베딩 (평가 서버 PPS_EMBED_DIR)
# ===================================================================================
# 조문 단위 분할. 예규(집행기준·낙찰자결정기준)는 장·절·번호 표제도 경계로 삼고,
# 각 청크에 (법령명, 조문키, 장·절 경로)를 붙여 항목표의 조문 인용을 결정론적으로 풀어낸다.
_LAW_ART = re.compile(r"^제\s?(\d+)조(?:의\s?(\d+))?\s*\(", re.M)
_LAW_CHAP = re.compile(r"^제\s?(\d+)\s?장\s*([^\n·]*)", re.M)
_LAW_SECT = re.compile(r"^제\s?(\d+)\s?절\s*([^\n·]*)", re.M)
_LAW_NUM_HEAD = re.compile(r"^(\d{1,2})\.\s+([^\n]{2,40})$", re.M)
_LAW_BYULPYO = re.compile(r"^\[?별표\s?(\d+)\]?", re.M)
_LAW_BUCHIK = re.compile(r"^부\s*칙", re.M)


class LawChunk:
    __slots__ = ("law", "key", "path", "text", "buchik")

    def __init__(self, law: str, key: str, path: str, text: str, buchik: bool):
        self.law, self.key, self.path, self.text, self.buchik = law, key, path, text, buchik


def _split_windows(text: str, max_chars: int) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    out, buf = [], ""
    for line in text.split("\n"):
        if len(buf) + len(line) + 1 > max_chars and buf:
            out.append(buf)
            buf = ""
        buf += line + "\n"
    if buf.strip():
        out.append(buf)
    return out


def load_law_chunks(data_dir: str, max_chars: int = 1200) -> List[LawChunk]:
    """법령패키지/법령/*.txt → 조문·표제 단위 청크(장·절 경로 포함)."""
    import glob
    out: List[LawChunk] = []
    seen: set = set()
    for path in sorted(glob.glob(os.path.join(data_dir, "법령패키지", "법령", "*.txt"))):
        law = os.path.splitext(os.path.basename(path))[0]
        try:
            txt = open(path, encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        # 경계 = 조문 / 장 / 절 / 'N. 표제' / 별표 / 부칙
        marks: List[Tuple[int, str, str]] = []           # (pos, kind, label)
        for m in _LAW_ART.finditer(txt):
            marks.append((m.start(), "art", f"제{m.group(1)}조" + (f"의{m.group(2)}" if m.group(2) else "")))
        for m in _LAW_CHAP.finditer(txt):
            marks.append((m.start(), "chap", f"제{m.group(1)}장 {m.group(2).strip()}"))
        for m in _LAW_SECT.finditer(txt):
            marks.append((m.start(), "sect", f"제{m.group(1)}절 {m.group(2).strip()}"))
        for m in _LAW_NUM_HEAD.finditer(txt):
            marks.append((m.start(), "num", f"{m.group(1)}. {m.group(2).strip()}"))
        for m in _LAW_BYULPYO.finditer(txt):
            marks.append((m.start(), "byul", f"별표{m.group(1)}"))
        for m in _LAW_BUCHIK.finditer(txt):
            marks.append((m.start(), "buchik", "부칙"))
        marks.sort()
        if not marks:
            for w in _split_windows(txt.strip(), max_chars):
                out.append(LawChunk(law, "", "", w.strip(), False))
            continue
        chap = sect = num = ""
        buchik = False
        bounds = [p for p, _, _ in marks] + [len(txt)]
        head = txt[: bounds[0]].strip()
        if len(head) > 200:                       # 표제부(법령 정보) 는 짧게 한 청크
            out.append(LawChunk(law, "", "", head[:600], False))
        for (pos, kind, label), nxt in zip(marks, bounds[1:]):
            body = txt[pos:nxt].strip()
            if kind == "chap":
                chap, sect, num = label, "", ""
            elif kind == "sect":
                sect, num = label, ""
            elif kind == "num":
                num = label
            elif kind == "buchik":
                buchik = True
            key = label if kind in ("art", "byul") else ""
            if kind == "chap" and chap and not buchik:
                pass
            if len(body) < 20:
                continue
            path_s = " > ".join(x for x in (chap, sect, num) if x)
            for w in _split_windows(body, max_chars):
                w = w.strip()
                sig = re.sub(r"\s+", "", w[:max_chars])
                if sig in seen:
                    continue
                seen.add(sig)
                out.append(LawChunk(law, key, path_s, w, buchik))
    return out


class LawRAG:
    """조문 청크의 bge-m3 임베딩 검색. PPS_EMBED_DIR 미설정/로드 실패 시 model=None(키워드 전용)."""

    def __init__(self, chunks: List[LawChunk], model=None, emb=None):
        self.chunks = chunks
        self.model = model
        self.emb = emb

    @classmethod
    def build(cls, data_dir: str) -> "LawRAG":
        chunks = load_law_chunks(data_dir)
        embed_dir = os.environ.get("PPS_EMBED_DIR")
        if not chunks:
            log("법령 색인 없음(법령패키지/법령/*.txt 미발견)")
            return cls(chunks)
        if not embed_dir:
            log(f"법령 색인 {len(chunks)}건 · 임베딩 생략(PPS_EMBED_DIR 미설정) → 조문 인용 결정론 매핑만 사용")
            return cls(chunks)
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer
            t0 = time.time()
            model = SentenceTransformer(embed_dir, device="cuda" if _cuda_ok() else "cpu")
            texts = [f"{c.law} {c.path} {c.text}" for c in chunks]
            emb = model.encode(texts, batch_size=32, normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=False)
            log(f"법령 RAG 준비: 조문 {len(chunks)}건 · {time.time() - t0:.1f}s")
            return cls(chunks, model, np.asarray(emb, dtype="float32"))
        except Exception as e:
            log(f"법령 RAG 생략(로드 실패 {type(e).__name__}: {e}) → 조문 인용 결정론 매핑만 사용")
            return cls(chunks)

    def semantic(self, query: str, top_k: int = 5, law: Optional[str] = None,
                 exclude: Optional[set] = None) -> List[int]:
        if self.model is None or self.emb is None:
            return []
        import numpy as np
        q = self.model.encode([query[:2000]], normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=False)[0]
        sims = self.emb @ np.asarray(q, dtype="float32")
        order = np.argsort(-sims)
        out: List[int] = []
        for i in order:
            i = int(i)
            c = self.chunks[i]
            if c.buchik or (law and c.law != law) or (exclude and i in exclude):
                continue
            out.append(i)
            if len(out) >= top_k:
                break
        return out

    def release(self) -> None:
        try:
            self.model = None
            import gc; gc.collect()
            import torch; torch.cuda.empty_cache()
        except Exception:
            pass


def _cuda_ok() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


# ===================================================================================
# 8-2. 항목 → 관련 조문 사전 매핑 (방안 A: 항목별 타겟 검색, 레코드마다 재검색 없음)
# ===================================================================================
# 항목표.json(data/항목표.json)의 국가/지방 조문 인용을 결정론적으로 풀고,
# bge-m3 가 있으면 항목 설명을 질의로 의미 검색해 보강한다. 두 법령체계(국가/지방)별로 1회 계산.
ITEM_TABLE: Dict[str, Dict[str, str]] = {
    "v1": {"항목명": "참가자격 특정기관 제한", "국가계약법": "국가계약법 시행령 제12조 국가계약법 시행령 제21조", "지방계약법": "지방계약법 시행령 제13조 지방계약법 시행령 제20조", "비고": ""},
    "v2": {"항목명": "고시금액 미만 실적제한", "국가계약법": "국가계약법 시행령 제21조 제1항 국가계약법 시행규칙 제25조", "지방계약법": "지방계약법 시행령 제20조 제1항 지방계약법 시행규칙 제25조 제2항 지방자치단체 입찰 및 계약 집행기준 제1장 입찰 및 계약 일반기준 제1절 총칙 7. 계약담당자 주의사항", "비고": "지방 + 소액수의 가능"},
    "v3": {"항목명": "실적제한 1배수 이상", "국가계약법": "국가계약법 시행령 제21조 제1항 국가계약법 시행규칙 제25조 (계약예규) 정부 입찰·계약 집행기준 제2장 제한경쟁입찰의 운용 제5조", "지방계약법": "지방계약법 시행령 제20조 제1항 지방계약법 시행규칙 제25조", "비고": "사업예산 기준"},
    "v4": {"항목명": "고시금액 이상 특정기관, 특정실적", "국가계약법": "(계약예규) 정부입찰계약집행기준 제2장 제한경쟁입찰의 운용 제5조", "지방계약법": "(행안부예규) 지방자치단체 입찰 및 계약 집행기준 제1장 입찰 및 계약 일반기준 7. 계약담당자 주의사항", "비고": "특정기관 표현 다양"},
    "v5": {"항목명": "고시금액 이상 지역제한", "국가계약법": "국가계약법 시행령 제21조 국가를 당사자로하는 계약에 관한 법률 등의 재정경제부장관이 정하는 고시금액", "지방계약법": "지방계약법 시행령 제20조 지방계약법 시행규칙 제24조", "비고": "지방, 지자체에 따라 고시금액 다름"},
    "v6": {"항목명": "고시금액 미만 지역제한 시,군,구", "국가계약법": "국가계약법 시행규칙 제25조", "지방계약법": "지방계약법 시행규칙 제25조 지방자치단체 입찰 및 계약집행기준 제5장 수의계약 운영요령 제3절 수의계약 대상과 운영요령 1. 금액기준에 따른 2인 이상 견적서 제출 수의계약", "비고": "지방 + 소액수의 가능"},
    "v7": {"항목명": "고시금액 미만 지역제한 인접 확대", "국가계약법": "국가계약법 시행규칙 제25조", "지방계약법": "지방계약법 시행규칙 제25조 지방자치단체 입찰 및 계약집행기준 제5장 수의계약 운영요령 제3절 수의계약 대상과 운영요령 1. 금액기준에 따른 2인 이상 견적서 제출 수의계약", "비고": "지방 + 소액수의 가능"},
    "v8": {"항목명": "중복제한 (실적+지역)", "국가계약법": "국가계약법 시행규칙 제25조", "지방계약법": "지방계약법 시행규칙 제25조 지방자치단체 입찰 및 계약집행기준 제5장 수의계약 운영요령 제3절 수의계약 대상과 운영요령 1. 금액기준에 따른 2인 이상 견적서 제출 수의계약", "비고": "지방 + 소액수의 가능"},
    "v9": {"항목명": "과업지시서 특정 모델명 명시", "국가계약법": "(계약예규) 정부 입찰·계약 집행기준 제2장 제한경쟁입찰의 운용 제5조", "지방계약법": "(행안부예규) 지방자치단체 입찰 및 계약 집행기준 제1장 입찰 및 계약일반기준 7. 계약담당자 주의사항", "비고": ""},
    "v10": {"항목명": "중기간 경쟁제품 입찰 직생 없음", "국가계약법": "중소기업제품 구매촉진 및 판로지원에 관한 법률 제9조 중소기업자간 경쟁제품 및 공사용자재 직접구매 대상 품목 지정 내역", "지방계약법": "중소기업제품 구매촉진 및 판로지원에 관한 법률 제9조 중소기업자간 경쟁제품 및 공사용자재 직접구매 대상 품목 지정 내역", "비고": ""},
    "v11": {"항목명": "중기간 경쟁제품 입찰 중소 없음", "국가계약법": "국가계약법 시행령 제21조 중소기업제품 구매촉진 및 판로지원에 관한 법률 제7조 제1항 중소기업자간 경쟁제품 및 공사용자재 직접구매 대상 품목 지정 내역", "지방계약법": "지방계약법 시행령 제20조 중소기업제품 구매촉진 및 판로지원에 관한 법률 제7조 제1항 중소기업자간 경쟁제품 및 공사용자재 직접구매 대상 품목 지정 내역", "비고": ""},
    "v12": {"항목명": "일반제품 직생 제한", "국가계약법": "국가계약법 시행령 제21조 중소기업제품 구매촉진 및 판로지원에 관한 법률 제9조", "지방계약법": "지방계약법 시행령 제20조 중소기업제품 구매촉진 및 판로지원에 관한 법률 제9조", "비고": "중기간경쟁제품 고시 참고"},
    "v13": {"항목명": "중기간 경쟁제품 소기업, 소상공인 제한", "국가계약법": "국가계약법 시행령 제21조 중소기업제품 구매촉진 및 판로지원에 관한 법률 제7조 제1항", "지방계약법": "지방계약법 시행령 제20조 중소기업제품 구매촉진 및 판로지원에 관한 법률 제7조 제1항", "비고": ""},
    "v14": {"항목명": "고시금액 이상 일반물품 중소기업 제한", "국가계약법": "국가계약법 시행령 제21조 중소기업제품 구매촉진 및 판로지원에 관한 법률 시행령 제2조의2 제2조의3", "지방계약법": "지방계약법 시행령 제20조 중소기업제품 구매촉진 및 판로지원에 관한 법률 시행령 제2조의2 제2조의3", "비고": ""},
    "v15": {"항목명": "1억 이상- 고시금액미만 소기업 제한", "국가계약법": "국가계약법 시행령 제21조 중소기업제품 구매촉진 및 판로지원에 관한 법률 시행령 제2조의2 제2조의3", "지방계약법": "지방계약법 시행령 제20조 중소기업제품 구매촉진 및 판로지원에 관한 법률 시행령 제2조의2 제2조의3", "비고": "판로지원 예외 명시한 경우, 제한 없어도 가능"},
    "v16": {"항목명": "1억 이상- 고시금액미만 중소기업 제한 없음", "국가계약법": "국가계약법 시행령 제21조 중소기업제품 구매촉진 및 판로지원에 관한 법률 시행령 제2조의2 제2조의3", "지방계약법": "지방계약법 시행령 제20조 중소기업제품 구매촉진 및 판로지원에 관한 법률 시행령 제2조의2 제2조의3", "비고": "판로지원 예외 명시한 경우, 제한 없어도 가능"},
    "v17": {"항목명": "1억원 미만 일반물품 중소기업 제한", "국가계약법": "국가계약법 시행령 제21조 중소기업제품 구매촉진 및 판로지원에 관한 법률 시행령 제2조의2 제2조의3", "지방계약법": "지방계약법 시행령 제20조 중소기업제품 구매촉진 및 판로지원에 관한 법률 시행령 제2조의2 제2조의3", "비고": "판로지원 예외 명시한 경우, 제한 없어도 가능"},
    "v18": {"항목명": "1억원 미만 일반물품 소기업 제한 없음", "국가계약법": "국가계약법 시행령 제21조 중소기업제품 구매촉진 및 판로지원에 관한 법률 시행령 제2조의2 제2조의3", "지방계약법": "지방계약법 시행령 제20조 중소기업제품 구매촉진 및 판로지원에 관한 법률 시행령 제2조의2 제2조의3", "비고": "판로지원 예외 명시한 경우, 제한 없어도 가능"},
    "v19": {"항목명": "물품공급 확약서 입찰 시 제출", "국가계약법": "국가계약법 시행령 제12조 국가계약법 시행규칙 제17조 (계약예규)정부 입찰·계약 집행기준 제2장 제한경쟁입찰의 운용 제5조의3", "지방계약법": "지방계약법 시행령 제13조 지방계약법 시행규칙 제17조 (행안부예규) 지방자치단체 입찰 및 계약 집행기준 제1장 입찰 및 계약일반기준 7. 계약담당자 주의사항", "비고": "입찰 전 발급, 계약시 제출 등 표현 다양"},
    "v20": {"항목명": "입찰참가자격 (SW) 대기업 참여제한 명시", "국가계약법": "소프트웨어 진흥법 제48조 중소 소프트웨어사업자의 사업 참여 지원에 관한 지침 제2조 별표1", "지방계약법": "소프트웨어 진흥법 제48조 중소 소프트웨어사업자의 사업 참여 지원에 관한 지침 제2조 별표1", "비고": "사업금액 기준 참여제한 다름 (20억 미만, 40억 미만, 80억 미만, 80억 이상)"},
    "v21": {"항목명": "공동수급 최소지분율 5% (10%)", "국가계약법": "(계약예규) 공동계약운용요령 제9조", "지방계약법": "(행안부예규) 지방자치단체 입찰 및 계약 집행기준 제6장 공동계약 운영요령", "비고": ""},
    "v22": {"항목명": "현장설명회 참석업체 자격 제한 (협상계약)", "국가계약법": "국가계약법 시행령 제43조", "지방계약법": "지방계약법 시행령 제43조", "비고": "설명회 참석을 참가자격으로 강제하는 근거 조항은 삭제되었다(국가 '19.12.18, 지방 '22.9.20)"},
    "v23": {"항목명": "협상계약 설명회~제안서 마감 기간 (지방)", "국가계약법": "", "지방계약법": "지방계약법 시행령 제35조 지방자치단체 입찰시 낙찰자 결정기준 제7장 협상에 의한 계약 낙찰자 결정기준 제3절 입찰과 계약상대자 결정절차", "비고": ""},
    "v24": {"항목명": "공고서와 나라장터 입력값 상이", "국가계약법": "", "지방계약법": "", "비고": "예산, 계약방법, 지역제한, 업종"},
}

# 인용 문자열의 법령명 별칭 → 법령패키지 파일명(확장자 제외). 공백·중점 제거 후 비교.
_LAW_ALIAS: List[Tuple[str, str]] = [
    ("국가계약법시행규칙", "국가를 당사자로 하는 계약에 관한 법률 시행규칙"),
    ("국가계약법시행령", "국가를 당사자로 하는 계약에 관한 법률 시행령"),
    ("국가를당사자로하는계약에관한법률등의재정경제부장관이정하는고시금액", "국가를 당사자로 하는 계약에 관한 법률 등의 재정경제부장관이 정하는 고시금액"),
    ("국가계약법", "국가를 당사자로 하는 계약에 관한 법률"),
    ("지방계약법시행규칙", "지방자치단체를 당사자로 하는 계약에 관한 법률 시행규칙"),
    ("지방계약법시행령", "지방자치단체를 당사자로 하는 계약에 관한 법률 시행령"),
    ("지방계약법", "지방자치단체를 당사자로 하는 계약에 관한 법률"),
    ("중소기업제품구매촉진및판로지원에관한법률시행규칙", "중소기업제품 구매촉진 및 판로지원에 관한 법률 시행규칙"),
    ("중소기업제품구매촉진및판로지원에관한법률시행령", "중소기업제품 구매촉진 및 판로지원에 관한 법률 시행령"),
    ("중소기업제품구매촉진및판로지원에관한법률", "중소기업제품 구매촉진 및 판로지원에 관한 법률"),
    ("중소기업자간경쟁제품및공사용자재직접구매대상품목지정내역", "중소기업자간 경쟁제품 및 공사용자재 직접구매 대상 품목 지정 내역"),
    ("정부입찰계약집행기준", "(계약예규) 정부 입찰·계약 집행기준"),
    ("지방자치단체입찰및계약집행기준", "지방자치단체 입찰 및 계약 집행기준"),
    ("지방자치단체입찰시낙찰자결정기준", "지방자치단체 입찰시 낙찰자 결정기준"),
    ("공동계약운용요령", "(계약예규) 공동계약운용요령"),
    ("중소소프트웨어사업자의사업참여지원에관한지침", "중소 소프트웨어사업자의 사업 참여 지원에 관한 지침"),
    ("소프트웨어진흥법시행령", "소프트웨어 진흥법 시행령"),
    ("소프트웨어진흥법", "소프트웨어 진흥법"),
]
_CIT_ART = re.compile(r"제\s?(\d+)\s?조(?:\s?의\s?(\d+))?")
_CIT_CHAP = re.compile(r"제\s?(\d+)\s?장")
_CIT_SECT = re.compile(r"제\s?(\d+)\s?절")
_CIT_NUM = re.compile(r"(?<!\d)(\d{1,2})\.\s*([가-힣][가-힣\s]{1,20})")
_CIT_BYUL = re.compile(r"별표\s?(\d+)")


def _norm_cit(s: str) -> str:
    return re.sub(r"[\s·ㆍ\(\)（）]", "", s).replace("계약예규", "").replace("행안부예규", "")


def parse_citations(cit: str) -> List[Tuple[str, str]]:
    """'국가계약법 시행령 제21조 제1항 국가계약법 시행규칙 제25조' → [(파일명, 그 법령에 딸린 인용 텍스트)]."""
    if not cit:
        return []
    n = _norm_cit(cit)
    hits: List[Tuple[int, int, str]] = []
    for alias, law in _LAW_ALIAS:
        start = 0
        while True:
            p = n.find(alias, start)
            if p < 0:
                break
            # 더 긴 별칭에 이미 포함된 위치는 건너뛴다
            if not any(a <= p < b for a, b, _ in hits):
                hits.append((p, p + len(alias), law))
            start = p + len(alias)
    hits.sort()
    out = []
    for i, (a, b, law) in enumerate(hits):
        end = hits[i + 1][0] if i + 1 < len(hits) else len(n)
        out.append((law, n[b:end]))
    return out


class ItemLawMap:
    """항목 × 법령체계(국가/지방) → 관련 조문 텍스트. 1회 계산 후 조회만 한다."""

    def __init__(self, rag: LawRAG, max_chars: int = 1800):
        self.rag = rag
        self.max_chars = max_chars
        self.ctx: Dict[Tuple[str, bool], str] = {}
        self.by_law: Dict[str, List[int]] = {}
        for i, c in enumerate(rag.chunks):
            self.by_law.setdefault(c.law, []).append(i)
        t0 = time.time()
        n_exact = n_sem = 0
        for v in ITEMS:
            for local in (False, True):
                idxs, ne, ns = self._resolve(v, local)
                n_exact += ne
                n_sem += ns
                self.ctx[(v, local)] = self._render(idxs)
        log(f"항목→조문 매핑 완료: 결정론 {n_exact} · 의미검색 {n_sem} · {time.time() - t0:.1f}s")

    def _find_article(self, law: str, key: str, chap: str = "") -> Optional[int]:
        cands = [i for i in self.by_law.get(law, []) if self.rag.chunks[i].key == key and not self.rag.chunks[i].buchik]
        if not cands:
            return None
        if chap:
            for i in cands:
                if chap in self.rag.chunks[i].path:
                    return i
        return cands[0]

    def _find_heading(self, law: str, tokens: List[str]) -> List[int]:
        """장·절·'N. 표제' 토큰이 경로에 모두 들어간 청크(본문 시작 2개)."""
        out = []
        for i in self.by_law.get(law, []):
            c = self.rag.chunks[i]
            if c.buchik or c.key:
                continue
            p = re.sub(r"\s+", "", c.path)
            if all(re.sub(r"\s+", "", t) in p for t in tokens):
                out.append(i)
                if len(out) >= 2:
                    break
        return out

    def _resolve(self, v: str, local: bool) -> Tuple[List[int], int, int]:
        row = ITEM_TABLE[v]
        cit = row["지방계약법"] if local else row["국가계약법"]
        idxs: List[int] = []
        n_exact = 0
        for law, rest in parse_citations(cit):
            if law not in self.by_law:
                continue
            chap_m = _CIT_CHAP.search(rest)
            chap = f"제{chap_m.group(1)}장" if chap_m else ""
            arts = [f"제{a}조" + (f"의{b}" if b else "") for a, b in _CIT_ART.findall(rest)]
            for k in arts:
                i = self._find_article(law, k, chap)
                if i is not None and i not in idxs:
                    idxs.append(i)
                    n_exact += 1
            for b in _CIT_BYUL.findall(rest):
                i = self._find_article(law, f"별표{b}")
                if i is not None and i not in idxs:
                    idxs.append(i)
                    n_exact += 1
            if not arts:                                   # 장·절·번호 표제 인용 (예규)
                toks = []
                if chap_m:
                    toks.append(f"제{chap_m.group(1)}장")
                sm = _CIT_SECT.search(rest)
                if sm:
                    toks.append(f"제{sm.group(1)}절")
                nm = _CIT_NUM.search(rest)
                if nm:
                    toks.append(f"{nm.group(1)}.{nm.group(2).strip()[:6]}")
                found = self._find_heading(law, toks) if toks else []
                if not found and not toks:                 # 조문 구조 없는 고시(지정 내역 등) → 파일 앞부분
                    found = [i for i in self.by_law[law] if not self.rag.chunks[i].buchik][:2]
                if not found and self.rag.model is not None:
                    found = self.rag.semantic(f"{row['항목명']} {cit}", top_k=2, law=law, exclude=set(idxs))
                for i in found:
                    if i not in idxs:
                        idxs.append(i)
                        n_exact += 1
        n_sem = 0
        if self.rag.model is not None and v != "v24":      # 의미 검색 보강(항목 설명 + 판정기준)
            q = f"{row['항목명']}. {ITEM_CRITERIA.get(v, '')} {row.get('비고', '')}"
            for i in self.rag.semantic(q, top_k=2, exclude=set(idxs)):
                idxs.append(i)
                n_sem += 1
        return idxs, n_exact, n_sem

    def _render(self, idxs: List[int]) -> str:
        parts, used = [], 0
        for i in idxs:
            c = self.rag.chunks[i]
            head = f"[{c.law}{(' ' + c.path) if c.path else ''}]"
            piece = f"{head}\n{c.text}"
            if used + len(piece) > self.max_chars:
                piece = piece[: max(0, self.max_chars - used)]
            if len(piece) < 40:
                break
            parts.append(piece)
            used += len(piece)
        return "\n\n".join(parts)

    def get(self, v: str, local: bool) -> str:
        return self.ctx.get((v, local), "")


# ===================================================================================
# 8-3. 2차 판정(캐스케이드) — 규칙이 애매한 (레코드, 항목) 쌍만 LLM 에 개별 질의
# ===================================================================================
# 1차 사실추출 프롬프트에 조문을 넣어도 LLM 출력은 거의 바뀌지 않았다(Colab 실측 188/200건 동일).
# 조문은 '위반 판정'을 시킬 때만 가치가 있으므로, 항목 정의·판정기준·관련 조문·공고 발췌를 묶어
# 항목 단위 예/아니오 판정을 별도로 묻고, 항목별 정책(억제/복구)으로 규칙 결과와 결합한다.
JUDGE_MAX_TOKENS = 360
JUDGE_MAX_PER_REC = 5
JUDGE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["why", "v", "conf", "q"],
    "properties": {
        "why": {"type": "string", "maxLength": 260},
        "v": {"type": "integer", "enum": [0, 1]},
        "conf": {"type": "string", "enum": ["high", "mid", "low"]},
        "q": {"type": ["string", "null"], "maxLength": 300},
    },
}

JUDGE_SYSTEM = """당신은 대한민국 공공조달 입찰공고의 법령 위반을 심사하는 계약심사 전문가다.
하나의 점검항목에 대해, 주어진 판정기준·관련 법령 조문·공고 발췌를 근거로 위반 여부를 판정한다.

원칙
- 판정기준에 적힌 조건(금액대·법령체계·계약방법·물품/용역 구분)을 그대로 적용한다. 조건이 하나라도 어긋나면 위반이 아니다.
- '참가자격'으로 요구된 것만 제한으로 본다. 제출서류 목록·평가기준·안내문구·상투적 문구(청렴계약, 부정당업자, 전자입찰 등록 등)는 제한이 아니다.
- 공고 발췌에 근거가 없는 사실을 추정하지 않는다. 근거가 없으면 v=0.
- q 는 위반의 근거가 되는 공고 원문 문장을 **글자 그대로** 복사한다(요약·의역 금지, 120자 이내). 위반이 아니면 null.
- conf: 근거 문장이 명확하고 판정기준에 정확히 부합하면 high, 해석이 필요하면 mid, 근거가 약하면 low.
- why 는 2문장 이내의 판정 이유.
JSON 객체 하나만 출력한다: {"why": ..., "v": 0|1, "conf": "high|mid|low", "q": ...}"""

# 항목별 판정기준(규칙 엔진 apply_rules 와 동일한 조건을 자연어로 명시)
ITEM_CRITERIA: Dict[str, str] = {
    "v1": "참가자격을 특정 기관·단체·법인 유형(대학·산학협력단·연구기관·협회·조합·재단·특정 단체 회원 등)으로만 한정했으면 위반. 업종·면허 등록, 지역제한, 기업규모 제한은 해당하지 않는다.\n위반 아님(예외): ①비영리법인·사회적기업·사회적협동조합·장애인기업·여성기업·중증장애인생산품시설을 '우대'하거나 가점을 주는 경우, ②관계 법령이 그 자격을 의무화한 경우(예: 폐기물처리업 허가, 문화재수리업 등록, 정보통신공사업 등록), ③해당 단체·법인도 참가할 수 있다고 '포함'을 밝힌 경우, ④입찰공고가 아닌 제출서류 목록·평가기준에 등장한 경우.",
    "v2": "추정가격이 고시금액(2억 3천만원) 미만인 물품·용역 입찰에서 과거 납품·수행 실적을 참가자격으로 요구했으면 위반. 지방계약법 + 소액수의(견적) 건은 예외.",
    "v3": "참가자격으로 요구한 실적 금액이 이 사업의 배정예산(추정가격×1.1 수준)을 초과하면 위반(실적 1배수 초과 요구). 예산과 같은 금액(1배)까지는 허용. 적격심사 평가기준·특별신인도의 실적 금액은 참가자격 제한이 아님. 실적 금액 요구 자체가 없으면 위반 아님.",
    "v4": "고시금액 이상 입찰에서 실적의 발주처·납품처를 특정 기관(국가기관·공공기관·지방자치단체·특정 병원 등)으로 한정했으면 위반. 발주처 제한 없는 유사실적 요구는 위반 아님.\n위반 아님(예외): ①'공공·민간 불문', '국내외', '발주처 제한 없음'이라고 밝힌 경우, ②적격심사·협상 평가기준(정량평가 배점표)의 실적 항목, ③제출서류 목록에 실적증명서가 있을 뿐 참가자격이 아닌 경우.",
    "v5": "추정가격이 지역제한 허용 상한(국가: 고시금액 2억 3천만원, 지방: 5억원) 이상인데 본점 소재지(지역)로 참가자격을 제한했으면 위반.",
    "v6": "추정가격이 지역제한 허용 상한 미만이고 지역제한을 하되, 광역(시·도)이 아닌 기초자치단체(시·군·구) 단위로 좁혔으면 위반. 지방 + 소액수의 건은 예외.",
    "v7": "추정가격이 지역제한 허용 상한 미만이고 지역제한을 둘 이상의 시·도(인접 시·도 포함)로 확대했으면 위반. 지방 + 소액수의 건은 예외.\n위반 아님(예외): ①제한 지역이 한 개 시·도(그 안의 시·군·구 나열 포함)뿐인 경우, ②열거된 지명이 참가자격이 아니라 납품장소·사업대상지·설명회 장소인 경우, ③지역제한 자체가 없는 경우, ④법령이 광역 확대를 허용한 경우(예: 세종·대전, 광주·전남 등 인접 지역 특례를 공고가 근거와 함께 밝힌 경우).",
    "v8": "지역제한과 실적제한을 동시에 참가자격으로 걸었으면(중복제한) 위반. 지방 + 소액수의 건은 예외.\n위반 아님(예외): ①실적이 참가자격이 아니라 적격심사·협상 평가기준(배점·가점)의 항목인 경우, ②실적증명서가 '제출서류 목록'에만 있고 보유 자체를 자격으로 요구하지 않은 경우, ③특별신인도·수행능력 평가의 실적, ④지역제한이 없고 실적만 있는 경우(또는 그 반대).",
    "v9": "규격서·과업지시서·공고문이 특정 제조사·상표·모델명(모델번호, 제품 시리즈)을 지정했으면 위반('동등 이상' 표기가 있어도 위반). 일반 규격·성능 수치만 있으면 위반 아님.\n위반 아님(예외): ①치수·용량·출력·해상도 등 일반 성능 수치와 KS/KC/CE/ISO 등 표준·인증 명칭만 있는 경우, ②나라장터 물품목록번호·세부품명번호·규격명 같은 조달 분류 코드, ③발주기관·납품장소·기존 설치 설비의 제조사를 '현황 설명'으로만 언급한 경우(호환 요구가 아님), ④소프트웨어 플랫폼·운영체제처럼 사실상 표준인 범용 명칭, ⑤'규격서에 기재된 성능(사양)과 동등 이상'처럼 성능 기준만 가리키는 문장.\n예시 — 위반: '○○사 Kymera 328i 또는 동등 이상', '모델명: SMK-3000', '제조사: ○○(주) 제품에 한함'. 위반 아님: 'CPU i7 이상, 메모리 16GB 이상', 'KS 인증 제품', '기존 설치된 ○○사 장비와 연동 가능할 것'(현황 설명).",
    "v10": "조달 대상이 중소기업자간 경쟁제품(중기부 지정 품목)인 용역·물품 입찰인데 직접생산확인증명서 보유를 참가자격으로 요구하지 않았으면 위반(부재). 직접생산확인 요구가 어디든 있으면 위반 아님.\n위반 아님(예외): ①'직접생산확인증명서', '직접생산 확인', '직접생산 여부 확인' 문구가 공고문·규격서·제출서류 어디에든 있는 경우, ②조달 대상이 경쟁제품이 아닌 경우, ③공사(건설)이거나 구매대행·리스 등 직접생산 개념이 적용되지 않는 경우.",
    "v11": "조달 대상이 중소기업자간 경쟁제품인데 참가자격에 중소기업(또는 소기업·소상공인) 제한이 전혀 없으면 위반(부재). 판로지원법 예외 명시가 있으면 위반 아님.\n위반 아님(예외): ①'중소기업자간 경쟁입찰', '중소기업제품 구매', '중소기업확인서 소지자' 등 중소기업 제한이 어떤 형태로든 있는 경우, ②판로지원법 시행령의 예외 사유를 밝힌 경우, ③조달 대상이 경쟁제품이 아닌 경우.",
    "v12": "조달 대상이 중소기업자간 경쟁제품이 아닌 일반 물품·용역인데 직접생산확인증명서 보유를 참가자격으로 요구했으면 위반.\n위반 아님(예외): ①조달 대상이 중소기업자간 경쟁제품이면 직접생산확인 요구가 적법하므로 위반이 아니다(세부품명번호가 경쟁제품 지정 목록에 있으면 경쟁제품이다), ②직접생산확인이 참가자격이 아니라 계약이행·납품검사 단계의 확인 절차인 경우, ③'직접생산' 문구가 제출서류 목록·안내문에만 있는 경우.",
    "v13": "조달 대상이 중소기업자간 경쟁제품인데 참가자격을 중기업을 배제하고 소기업·소상공인으로만 한정했으면 위반(경쟁제품은 중소기업 전체가 참여 가능해야 한다).\n위반 아님(예외): ①'중소기업'(중기업 포함)으로 적혀 있거나 '중소기업자간 경쟁입찰'이라고만 한 경우('중소기업 또는 소상공인', '중·소기업·소상공인 확인서' 병기도 중기업이 포함되므로 여기), ②소기업·소상공인이 적격심사 배점표·가점·우대 대상일 뿐 참가 제한 문장이 아닌 경우, ③조달 대상이 경쟁제품이 아닌 경우, ④소기업·소상공인 문구가 제출서류·안내문에만 있는 경우.\n예시 — 위반: 경쟁제품(직접생산확인증명서 요구) + '소기업 또는 소상공인으로서 소기업·소상공인 확인서를 소지한 자'. 위반 아님: 경쟁제품 + '중소기업자로서 중소기업확인서를 소지한 자', 또는 배점표의 '소기업·소상공인 가점 2점'.",
    "v14": "경쟁제품이 아닌 일반 물품·용역에서 추정가격이 고시금액(2억 3천만원) 이상인데 참가자격을 중소기업(또는 소기업·소상공인)으로 제한했으면 위반.\n위반 아님(예외): ①조달 대상이 중소기업자간 경쟁제품인 경우, ②중소기업이 '우대·가점' 대상일 뿐 참가 제한이 아닌 경우, ③추정가격이 고시금액 미만인 경우, ④중소기업확인서가 제출서류 목록에만 있고 자격 제한 문장이 없는 경우, ⑤공사(건설)이거나 물품·용역이 아닌 경우.",
    "v15": "경쟁제품이 아닌 일반 물품·용역에서 추정가격이 1억원 이상 고시금액 미만인데 참가자격을 소기업·소상공인으로만 한정했으면 위반(이 구간은 중소기업 전체 제한이 맞다).\n위반 아님(예외): ①'중소기업'(중기업 포함)까지 참가할 수 있게 한 경우('중소기업 또는 소상공인' 병기 포함), ②소기업·소상공인이 우대·가점 대상일 뿐인 경우, ③경쟁제품인 경우, ④추정가격이 이 구간을 벗어난 경우, ⑤'소기업·소상공인 확인서'가 제출서류 목록에만 있는 경우.",
    "v16": "경쟁제품이 아닌 일반 물품·용역의 경쟁입찰에서 추정가격이 1억원 이상 고시금액 미만인데 참가자격에 중소기업 제한이 전혀 없으면 위반(부재). 판로지원법 시행령 제2조의3 예외 명시 또는 수의계약이면 위반 아님.\n위반 아님(예외): ①'중소기업확인서를 소지한 자', '소기업·소상공인 확인서 소지자' 등 규모 제한이 어떤 형태로든 있는 경우(소기업·소상공인 제한도 중소기업 제한에 포함된다), ②수의계약·소액수의(견적)인 경우, ③경쟁제품인 경우, ④판로지원법 시행령 제2조의3 예외를 밝힌 경우, ⑤공사(건설)인 경우.",
    "v17": "경쟁제품이 아닌 일반 물품·용역에서 추정가격이 1억원 미만인데 참가자격을 (소기업·소상공인이 아닌) 중소기업 전체로 제한했으면 위반(이 구간은 소기업·소상공인 제한이 맞다).\n위반 아님(예외): ①자격 문장이 '소기업 또는 소상공인으로서 소기업·소상공인 확인서를 소지한 자'처럼 중기업을 배제한 경우, ②중소기업이 우대·가점 대상일 뿐인 경우, ③경쟁제품인 경우, ④추정가격이 1억원 이상인 경우, ⑤중소기업확인서가 제출서류 목록에만 있는 경우.\n주의: '중소기업 또는 소상공인', '중·소기업·소상공인 확인서를 소지한 업체' 병기는 중기업을 포함하므로 중소기업 전체 제한 = 위반이다(소기업·소상공인만 적은 경우와 구별).\n예시 — 위반: 추정가격 5천만원 용역 + '「중소기업기본법」 제2조에 따른 중소기업 또는 소상공인으로서 중소기업·소상공인 확인서를 소지한 업체'. 위반 아님: 같은 금액 + '소기업 또는 소상공인으로서 소기업·소상공인 확인서를 소지한 자'.",
    "v18": "경쟁제품이 아닌 일반 물품·용역의 경쟁입찰에서 추정가격이 1억원 미만인데 참가자격에 소기업·소상공인 제한이 없으면 위반(부재). 판로지원법 예외 명시 또는 수의계약이면 위반 아님.\n위반 아님(예외): ①'소기업·소상공인 확인서 소지자'뿐 아니라 '중소기업확인서 소지자' 등 규모 제한이 어떤 형태로든 있는 경우, ②수의계약·소액수의(견적)인 경우, ③경쟁제품인 경우, ④판로지원법 예외를 밝힌 경우, ⑤공사(건설)인 경우.",
    "v19": "물품 공고에서 제조사·공급사의 물품공급(기술지원·A/S) 확약서·공급증명원을 입찰참가 시점에 요구했으면 위반 — '입찰서 제출 마감(전일)까지 보유·발급·제출', '미보유 시 입찰 제외', 입찰 참가 제출서류로 확약서 요구, 또는 마감 전 보유를 요구하면서 제출만 계약 시로 미룬 경우.\n위반 아님(예외): ①마감 전 보유 요구 없이 계약 체결 시·낙찰 후·적격심사 시·납품(검수) 시에만 제출하도록 한 경우, ②청렴서약서·입찰보증금 납부이행 확약처럼 물품공급 확약이 아닌 것, ③용역·공사 공고인 경우.",
    "v20": "소프트웨어사업(개발·정보시스템 구축·운영·유지관리)인데 공고문·제안요청서에 대기업(중견기업) 소프트웨어사업자 참여제한 적용 여부를 명시하지 않았으면 위반(부재).\n위반 아님(예외): ①'대기업인 소프트웨어사업자는 참여할 수 없다' 또는 '참여제한 대상 아님/예외 인정'처럼 적용 여부를 어떤 형태로든 밝힌 경우, ②소프트웨어 사업이 아닌 경우(단순 물품 구매, 하드웨어 납품, SW 라이선스 단순 구매, 교육·연구용역 등).",
    "v21": "공동수급체 구성원별 최소지분율을 국가계약 10% 미만(예: 5%) 또는 지방계약 5% 미만으로 정했으면 위반. 지분율 명시가 없으면 위반 아님.",
    "v22": "협상에 의한 계약에서 현장(사업·제안요청)설명회 참석을 입찰(제안서 제출) 자격 요건으로 강제했으면 위반(미참석 업체 참가 불가). 설명회를 단순 개최만 하면 위반 아님.",
    "v23": "지방계약법 + 협상에 의한 계약에서 제안요청(현장)설명회 개최일부터 제안서 제출 마감일까지의 기간이 기준(추정가격 10억 이상 40일, 1억 이상 20일, 1억 미만 10일)보다 짧으면 위반.\n위반 아님(예외): ①설명회 개최일 또는 제안서 마감일이 공고문에 명시되지 않아 기간을 셀 수 없는 경우, ②국가계약법 적용 건, ③협상에 의한 계약이 아닌 경우, ④재공고·긴급공고로 단축 근거를 밝힌 경우.",
    "v24": "공고문 본문이 밝힌 예산·추정가격, 계약방법(일반경쟁·제한경쟁·수의), 지역제한 대상 지역, 업종·면허 중 하나라도 나라장터 입력값(메타)과 다르면 위반. 메타가 비어 있으면 위반 아님.\n위반 아님(예외): ①금액 차이가 부가가치세 포함/제외(추정가격×1.1) 또는 단위 표기 차이로 설명되는 경우, ②메타의 계약방법 문구가 공고문 어디엔가 그대로 있는 경우, ③공고문 값이 메타보다 상세할 뿐 모순되지 않는 경우(예: 메타 '제한경쟁' + 공고 '지역제한 경쟁입찰'), ④메타 항목이 비어 있거나 '미기재'인 경우.",
}

# 항목별 발췌 패턴(공고 본문에서 판정에 필요한 문장만 뽑는다)
ITEM_EXCERPT: Dict[str, str] = {
    "v1": r"참가\s*자격|기관|단체|법인|협회|조합|대학|재단",
    "v2": r"실적", "v3": r"실적", "v4": r"실적|발주|납품처",
    "v5": r"지역\s*제한|본점|주된\s*영업소|소재지", "v6": r"지역\s*제한|본점|주된\s*영업소|소재지",
    "v7": r"지역\s*제한|본점|주된\s*영업소|소재지|인접", "v8": r"지역\s*제한|본점|주된\s*영업소|소재지|실적",
    "v9": r"모델\s*명|모델명|모델|모델\s*번호|품번|제품명|제조사|제조업체|제조원|상표|브랜드|동등\s*이상|규격|품명|시리즈|[Mm]odel|[Bb]rand",
    "v10": r"직접\s*생산|경쟁제품|세부\s*품명|중소기업자간", "v11": r"중소기업|소기업|소상공인|경쟁제품|예외|확인서|규모",
    "v12": r"직접\s*생산|경쟁제품|세부\s*품명", "v13": r"소기업|소상공인|중소기업|경쟁제품|확인서|직접\s*생산|가점|우대",
    "v14": r"중소기업|소기업|소상공인", "v15": r"소기업|소상공인|중소기업", "v16": r"중소기업|소기업|소상공인|예외|우선조달",
    "v17": r"중소기업|소기업|소상공인|확인서|기업\s*규모|규모\s*제한|가점|우대", "v18": r"중소기업|소기업|소상공인|예외|우선조달|확인서",
    "v19": r"확약서|공급\s*확약|공급\s*증명|기술\s*지원|공급\s*협약|제조사",
    "v20": r"대기업|중견기업|소프트웨어사업자|상호출자제한|참여\s*제한|소프트웨어",
    "v21": r"지분율|출자\s*비율|참여\s*비율|공동\s*수급|공동\s*계약|공동\s*도급",
    "v22": r"설명회|현장\s*설명|참석", "v23": r"설명회|현장\s*설명|제안서\s*(?:제출|접수)|제출\s*마감|접수\s*마감|개찰",
    "v24": r"추정\s*가\s*격|배정\s*예산|사업\s*예산|계약\s*방법|입찰\s*방법|지역\s*제한|업종|면허",
}

# 2차 판정 우선순위(dev200 F1 낮은 항목 먼저 — 데드라인 절단 시 가치 높은 질의가 먼저 처리된다)
JUDGE_PRIORITY = ["v10", "v13", "v24", "v17", "v9", "v12", "v18", "v8", "v11", "v16", "v4", "v23",
                  "v14", "v2", "v1", "v15", "v20", "v6", "v19", "v7", "v5", "v3", "v21", "v22"]

# 결합 정책. suppress: 규칙 1 → LLM(v=0, conf≥기준)이면 0. recover: 규칙 0 → LLM(v=1, conf≥기준, 인용 원문일치)이면 1.
# 정밀도 1.0 인 항목(v1·v3·v5·v9·v12·v19·v21·v22)은 억제하지 않고, FP 과다 항목은 복구하지 않는다.
_CONF_RANK = {"low": 0, "mid": 1, "high": 2}
# [변경] Colab 실측(dev200 judge.jsonl 599질의 + 무라벨300 803질의)으로 항목별 판정 신뢰도를 검증한 결과:
#   - 복구(규칙0→판정1, high)는 거의 전 항목에서 정밀도 10~20%(v9 1/9, v19 1/11, v13 1/5, v15 0/4, v17 0/3, v20 0/3).
#     무라벨 300건에서는 v13 24건·v9 12건·v19 12건이 복구될 판이라(test 8%/4%/4%) 그대로 두면 test FP 폭증.
#     → 복구는 v1(dev 1/0, 무라벨 3/300, 인용문 검토 시 타당)만 남긴다.
#   - 억제(규칙1→판정0, high)는 항목별로 갈린다: v8 2/0·v11 3/0·v18 1/0·v23 1/0·v24 1/0 은 맞고,
#     v1 0/3·v6 0/2·v10 4/3·v4 2/2·v16 0/1 은 틀리거나 반반 → 검증된 5개만 남긴다.
#   기본 정책(전 항목 high) dev200 0.7344(규칙 0.7726 대비 -0.038) → 이 정책 재생 결과는 아래 리포트 참조.
JUDGE_POLICY: Dict[str, Dict[str, Any]] = {
    # ── [원칙 2026-09-16] 프록시(자가라벨 1,984건)·Colab 실측 없이 정책 추가 금지 ──
    #  경위: dev200(항목당 양성 ~6건) 기준으로 2026-09-10 에 v4 recover + v7/v13/v14/v15/v17/v20
    #  suppress 를 추가했으나 LB 0.6596→0.6511 로 하락. 해당 정책은 프록시·GPU 실측 어느 쪽으로도
    #  검증되지 않은 채 들어간 것이어서 전부 제거한다. 아래에 남은 항목은 2026-09-09 Colab 실측
    #  (dev200 공식 정답, kit 9865aba0)에서 효과가 확인된 것만이다.
    #  [재도입하지 않는 것 — 실측 반증 있음]
    #   · v9/v19 recover: dev 실측 v9 FP +5(F1 0.80→0.53), v19 FP +10(0.91→0.55). 자가라벨도 모델명·확약서 과잉 위반.
    #   · v10/v4/v16/v6/v1 suppress: dev 실측 v10 4/3·v4 2/2·v16 0/1·v6 0/2·v1 0/3.
    #   · v13/v15/v17/v20 recover: dev 실측 정밀도 1/5·0/4·0/3·0/3.
    #   · v4 recover, v7/v13/v14/v15/v17/v20 suppress: 미실측 상태로 투입 → LB 하락 동반(2026-09-10 제출).
    #  ── [2026-09-16b] 동일 계열 Gemma(GEMMA_4_26B, API)로 dev200·프록시 1,984건에 사실추출+판정 풀 파이프라인을
    #  실측(eval_api.py → policy_lab.py). 규칙만 dev 0.876/프록시 0.801 → 판정(구 정책) 0.845/0.772 로
    #  판정 단계가 -0.03 을 깎고 있었다. 원인은 suppress:high 5개 중 4개:
    #   · v11 suppress: dev 1.000→0.500(2/0/4). "직접생산증명서 요구=중소기업 제한 있음"으로 오판(DEV-060/063/064).
    #   · v18 suppress: dev 0.800→0.615. "제출서류에 소기업확인서 있음=제한 있음"으로 오판(DEV-038/043).
    #   · v8 suppress: 프록시 0.708→0.432(8/2/19), dev 0.833→0.769.  · v23 0.571→0.333  · v24 0.733→0.702.
    #  항목별 9조합 탐색(기준: 프록시 +0.01 이상 & dev 하락 없음) 결과 아래 정책으로 dev 0.845→0.869, 프록시 0.772→0.803.
    #  (2026-09-09 Colab 실측과 다른 이유: 당시보다 ITEM_CRITERIA 예외절이 확장되어 v11/v18 판정이 흔들림.
    #   구 예외절로 되돌리면 v11/v18 은 회복되지만 v1 recover 가 0.923→0.750 으로 무너져 신 예외절+본 정책을 택함.)
    "v1": {"recover": "high"},
    "v8": {"recover": "mid"},
    "v15": {"suppress": "mid"},
    "v19": {"suppress": "mid"},
    "v20": {"recover": "mid"},
}


def load_policy_asset(asset_dir: str = ASSET_DIR) -> Optional[str]:
    """model/policy_margin.json 이 있으면 JUDGE_POLICY 를 그 정적 자산으로 교체한다.

    이 파일은 Colab(vLLM) 에서 수집한 v 토큰 logprob(ln-odds)을 policy_lab.py 로
    dev200(공식 정답)에 대해 전수탐색해 만든 '튜닝된 판정 임계 정책'이다.
    가중치가 아니라 정적 설정 파일이므로 대회 규정상 model/ 에 넣을 수 있다.
    형식(둘 다 허용):
      {"policy": {"v1": {"recover":"high","recover_margin":2.0}, ...}, "_meta": {...}}
      {"v1": {"recover":"high","recover_margin":2.0}, ...}
    파일이 없거나 비면 내장 JUDGE_POLICY 를 그대로 쓴다(회귀 0). 반환: 적용한 경로 또는 None.
    """
    p = os.path.join(asset_dir, "policy_margin.json")
    if not os.path.isfile(p):
        return None
    try:
        data = json.load(io.open(p, encoding="utf-8"))
        pol = data.get("policy", data) if isinstance(data, dict) else {}
        clean = {v: spec for v, spec in pol.items()
                 if v in ITEMS and isinstance(spec, dict)}
        if not clean:
            log(f"정책 자산 {p} 에 유효 항목 없음 → 내장 JUDGE_POLICY 사용")
            return None
        JUDGE_POLICY.clear()
        JUDGE_POLICY.update(clean)
        return p
    except Exception as e:
        log(f"정책 자산 로드 실패({type(e).__name__}: {e}) → 내장 JUDGE_POLICY 사용")
        return None


_RE_ANY_PERF = re.compile(r"실적")
_RE_ANY_SIZE = re.compile(r"중소기업|소기업|소상공인")
_RE_ANY_INST = re.compile(_INST_TERM)


def judge_candidates(res: Dict[str, Tuple[int, str]], f: Dict[str, Any],
                     rec: Dict[str, Any], pre: Dict[str, Any]) -> List[str]:
    """2차 판정 대상 항목: 규칙이 발화한 항목(검증) + 약한 게이트만 걸린 항목(복구 후보)."""
    ft = flat_text(rec)
    qual = pre.get("qual") or ""
    est = pre["est"]
    local = pre["local"]
    band = pre["band"]
    thr = region_threshold(local)
    nego = is_negotiation(rec)
    smallq = is_small_quote(rec)
    local_smallq = local and smallq
    fired = {v for v in ITEMS if res.get(v, (0, ""))[0] == 1}
    reg = bool(pre.get("reg_det")) or f.get("reg") == 1
    perf_w = bool(_RE_ANY_PERF.search(qual))
    cmp_any = bool(f.get("cmp") == 1 or pre.get("comp_code_hit") or pre.get("comp_name_hit")
                   or pre.get("comp_joh") or pre.get("comp_panro9"))
    svc = not pre.get("is_goods", False)
    weak: Dict[str, bool] = {
        "v1": bool(re.search(r"(?:%s)[^\n]{0,25}(?:만|한함|한정|이어야|에\s*한|참가\s*가능)" % _INST_TERM, qual)),
        "v2": perf_w and est < GOSI_NATIONAL and not local_smallq,
        "v3": perf_w and bool(re.search(r"실적[^\n]{0,60}(?:원|억|천만)", qual)),
        "v4": perf_w and bool(re.search(_PUB_ORG + r"|발주|납품처|병원", qual)),
        "v5": reg and est >= thr,
        "v6": reg and est < thr and not local_smallq and bool(re.search(r"시|군|구", pre.get("reg_quote") or "")),
        "v7": reg and est < thr and not local_smallq and bool(re.search(r"인접|또는|및|,", pre.get("reg_quote") or "")),
        "v8": reg and perf_w and not local_smallq,
        "v9": bool(re.search(r"모델\s*명|모델명|제조사|상표|동등\s*이상|브랜드", ft)),
        "v10": cmp_any and svc,
        "v11": cmp_any and svc,
        "v12": (not cmp_any) and bool(_RE_DPC_ANY.search(ft)),
        "v13": cmp_any and svc and bool(re.search(r"소기업|소상공인", qual)),
        "v14": (not cmp_any) and band == "over_gosi" and bool(_RE_ANY_SIZE.search(qual)),
        "v15": (not cmp_any) and band == "mid" and bool(re.search(r"소기업|소상공인", qual)),
        "v16": False,                                    # 부재탐지 → 발화분 검증만
        "v17": (not cmp_any) and band == "under1e" and bool(re.search(r"중소기업", qual)),
        "v18": False,
        "v19": pre.get("is_goods", False) and bool(re.search(r"확약서|공급\s*확약|공급\s*증명|공급\s*확인", ft)),
        "v20": bool(pre.get("sw_det")) or bool(re.search(r"소프트웨어\s*사업|정보시스템\s*구축|시스템\s*유지관리", ft)),
        "v21": bool(pre.get("share_hint")) or bool(re.search(r"지분율|출자\s*비율", ft)),
        "v22": nego and "설명회" in ft,
        "v23": local and nego and "설명회" in ft,
        "v24": False,
    }
    # [변경] 정책이 없는 항목은 판정 결과가 결과에 반영되지 않으므로 질의하지 않는다(질의 수 ≈ 1/3, 런타임 절감).
    #        발화 항목은 suppress 정책, 미발화 항목은 recover 정책이 있을 때만 후보.
    cands = [v for v in JUDGE_PRIORITY
             if (v in fired and "suppress" in JUDGE_POLICY.get(v, {}))
             or (v not in fired and weak.get(v) and "recover" in JUDGE_POLICY.get(v, {}))]
    return cands[:JUDGE_MAX_PER_REC]


# 참가자격 블록에서 평가기준·배점·가점·제출서류 성격의 줄을 가려내는 패턴.
# 줄 전체가 아니라 '제한 문장'이 아닌 부가 문구를 겨냥한다: 배점표 헤더, 가점·우대 안내, 제출서류 열거(…1부/…각 1부).
_RE_QUAL_SIDE = re.compile(
    r"배점|평가\s*기준|평가\s*항목|정량\s*평가|가\s*점|우대|만점|점수|\d+\s*점\b"
    r"|(?:각\s*)?\d+\s*부\s*$|제출\s*서류|첨부\s*서류|증명서\s*1부|확인서\s*1부")
_RE_QUAL_LIMIT = re.compile(r"소지한|이어야|한함|한정|에\s*한|참가할\s*수|참가\s*가능|참가\s*불가|없습니다|제외")


_JUDGE_SPLIT_ITEMS = {"v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
                      "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18"}


def split_qual_side_lines(qual: str) -> Tuple[str, str]:
    """참가자격 블록을 (제한 문장 블록, 평가·배점·제출서류 줄 블록)으로 나눈다.

    '소기업·소상공인 확인서를 소지한 자'처럼 제한 동사가 함께 있는 줄은 배점 단어가 있어도 본문에 남긴다.
    """
    core, side = [], []
    for line in qual.split("\n"):
        t = line.strip()
        if t and _RE_QUAL_SIDE.search(t) and not _RE_QUAL_LIMIT.search(t):
            side.append(t)
        else:
            core.append(line)
    return "\n".join(core).strip(), "\n".join(side).strip()


def build_judge_messages(v: str, rec: Dict[str, Any], pre: Dict[str, Any],
                         law_ctx: str) -> List[Dict[str, str]]:
    nt = notice_text(rec)
    st = spec_text(rec)
    row = ITEM_TABLE[v]
    est = pre["est"]
    facts = [
        f"추정가격 {est:,}원 · 배정예산 {pre['budget']:,}원 · 금액대 "
        + {"over_gosi": "고시금액(2.3억) 이상", "mid": "1억 이상~고시금액 미만", "under1e": "1억원 미만"}.get(pre["band"], pre["band"]),
        f"적용 법령체계: {'지방계약법' if pre['local'] else '국가계약법'} · 계약방법: {meta_str(rec, '계약방법') or '미기재'}"
        f" · 낙찰방법: {meta_str(rec, '낙찰방법') or '미기재'} · 업무구분: {meta_str(rec, '업무구분') or '미기재'}",
        f"지역제한 허용 상한: {region_threshold(pre['local']):,}원 · 공동수급 최소지분율 기준: {MIN_SHARE_LOCAL if pre['local'] else MIN_SHARE_NATIONAL:g}%",
    ]
    if pre.get("comp_codes"):
        facts.append("경쟁제품 세부품명번호 일치: " + ", ".join(pre["comp_codes"][:4]))
    elif pre.get("comp_names") and v in ("v10", "v11", "v12", "v13"):
        facts.append("경쟁제품 유사 품목명 후보(확정 아님): " + ", ".join(pre["comp_names"][:4]))
    if pre.get("comp_joh") and v in ("v10", "v11", "v12", "v13"):
        # 나라장터 '조항호내용'(계약 근거조항)이 경쟁제품 지정을 직접 밝힌다 — 품명 후보와 달리 확정 신호(dev 061).
        facts.append("나라장터 조항호내용(계약 근거): " + clip(meta_str(rec, "조항호내용"), 80) + " → 중기간 경쟁제품 공고로 확정")
    if pre.get("comp_excluded"):
        facts.append("주의: 경쟁제품 목록에 있으나 추정가격 조건을 벗어나 이 건에서는 경쟁제품이 아님: " + ", ".join(pre["comp_excluded"][:2]))
    if v in ("v22", "v23"):
        if pre.get("brief_day"):
            facts.append("규칙검출 설명회 개최일 %04d-%02d-%02d" % pre["brief_day"])
        if pre.get("prop_day"):
            facts.append("규칙검출 제안서 마감일 %04d-%02d-%02d" % pre["prop_day"])
        if v == "v23":
            facts.append(f"이 건의 최소 요구 기간: {required_brief_days(est)}일")
    parts = [
        f"## 점검항목 {v}: {row['항목명']}",
        "## 판정기준\n" + ITEM_CRITERIA[v] + (f"\n(참고: {row['비고']})" if row.get("비고") else ""),
    ]
    if law_ctx:
        parts.append("## 관련 법령 조문\n" + law_ctx)
    parts.append("## 이 공고의 확정 사실(나라장터 입력값 기반)\n" + "\n".join(f"- {x}" for x in facts))
    if v == "v24":
        parts.append("## 나라장터 입력값(메타) 전체\n" + format_meta_block(rec))
    parts.append("## 공고 개요\n" + notice_head(nt, 500))
    qual = pre.get("qual") or ""
    if qual:
        # [추가] 참가자격 블록 안에 섞인 평가기준·배점·가점·제출서류 줄을 분리해 보여준다 — 판정 모델이
        #        '실적 배점 30점'·'소기업 가점'·'확인서 1부' 같은 줄을 참가자격 제한으로 오독하는 것을 막는다
        #        (자가라벨 v13/v17 FP 사유 다수가 이 유형). 원문 순서는 두 블록 각각에서 유지한다.
        #        v19 는 제출서류 목록의 '확약서 1부'가 곧 근거(dev 033 정답 1)이므로 분리하지 않는다.
        q_core, q_side = split_qual_side_lines(qual) if v in _JUDGE_SPLIT_ITEMS else (qual, "")
        parts.append("## 입찰참가자격\n" + clip(q_core, 1600))
        if q_side:
            parts.append("## 참가자격 조항 안의 평가·배점·제출서류 줄(제한 문장과 구분해 보인 것 — 자격 '요건'인지 확인할 때만 참고)\n"
                         + clip(q_side, 500))
    src = (st + "\n" + nt) if v == "v9" else (nt + "\n" + st if v in ("v19", "v20") else nt)
    exc = extract_around(src, ITEM_EXCERPT[v], 360, 4, 1500)
    if exc:
        parts.append("## 관련 발췌\n" + exc)
    parts.append(f"위 자료만으로 {v}({row['항목명']}) 위반 여부를 판정하라.")
    return [{"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": "\n\n".join(parts)}]


def parse_judge(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    s = text.strip()
    a, b = s.find("{"), s.rfind("}")
    if a < 0 or b <= a:
        return None
    try:
        d = json.loads(s[a:b + 1])
    except Exception:
        return None
    if not isinstance(d, dict) or "v" not in d:
        return None
    try:
        vv = int(d.get("v"))
    except Exception:
        return None
    conf = str(d.get("conf") or "low").lower()
    if conf not in _CONF_RANK:
        conf = "low"
    q = d.get("q")
    q = str(q).strip() if isinstance(q, str) and q.strip() else None
    return {"v": 1 if vv == 1 else 0, "conf": conf, "q": q, "why": clip(str(d.get("why") or ""), 260)}


_CONF_NAME = ["low", "mid", "high"]


def _demote(conf: str, steps: int = 1) -> str:
    return _CONF_NAME[max(0, _CONF_RANK.get(conf, 0) - steps)]


def merge_judge_samples(samples: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """[추가] F) 자기일관성 다수결 + G) logprob 기권을 한 번에 적용한다.

    · v(0/1) 다수결. 만장일치가 아니면 conf 를 한 단계 낮춘다(2/3 → mid → 정책 미발동).
    · 다수파 안에서 conf 가 가장 높은 샘플의 인용문·사유를 대표로 쓴다.
    · v 토큰 ln-odds 중앙값이 JUDGE_MARGIN_MIN 미만이면 conf 를 low 로 강등(기권).
    · 파싱 성공 샘플이 하나도 없으면 None(=판정 실패, 규칙 결과 유지).
    """
    parsed: List[Tuple[Dict[str, Any], Optional[float]]] = []
    for s in samples or []:
        j = parse_judge(s.get("text", "") if isinstance(s, dict) else str(s))
        if j is not None:
            parsed.append((j, s.get("margin") if isinstance(s, dict) else None))
            if isinstance(s, dict) and s.get("lo") is not None:
                j["_lo"] = float(s["lo"])
    if not parsed:
        return None
    n1 = sum(1 for j, _ in parsed if j["v"] == 1)
    win = 1 if n1 * 2 > len(parsed) else 0
    if n1 * 2 == len(parsed):                          # 동수(짝수 샘플) → 보수적으로 0
        win = 0
    grp = [(j, m) for j, m in parsed if j["v"] == win]
    grp.sort(key=lambda t: (_CONF_RANK.get(t[0]["conf"], 0), 1 if t[0].get("q") else 0), reverse=True)
    best = dict(grp[0][0])
    n_agree = len(grp)
    if n_agree < len(parsed):                          # 샘플 간 불일치 → 신뢰도 강등
        best["conf"] = _demote(best["conf"])
    ms = [m for _, m in grp if m is not None]
    if ms:
        ms.sort()
        med = ms[len(ms) // 2]
        if med < JUDGE_MARGIN_MIN:                     # 결정 토큰이 애매 → 기권
            best["conf"] = "low"
        best["margin"] = round(float(med), 3)
    los = sorted(j.get("_lo") for j, _ in grp if j.get("_lo") is not None)
    if los:
        best["lo"] = round(float(los[len(los) // 2]), 3)   # 부호 있는 ln-odds(임계 튜닝용)
    best.pop("_lo", None)
    best["n"] = len(parsed)
    best["agree"] = n_agree
    return best


def apply_judgments(res: Dict[str, Tuple[int, str]], jd: Dict[str, Dict[str, Any]],
                    idx: "EvidenceIndex", policy: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Tuple[int, str]]:
    """규칙 결과에 2차 판정을 항목별 정책으로 결합한다."""
    pol = policy if policy is not None else JUDGE_POLICY
    out = dict(res)
    for v, j in (jd or {}).items():
        if v not in out or not j:
            continue
        hit, quote = out[v]
        p = pol.get(v, {})
        rank = _CONF_RANK.get(j.get("conf", "low"), 0)
        # [추가 2026-09-20] logprob 임계: 정책에 suppress_margin / recover_margin 이 있으면 v 토큰
        #   ln-odds 절대값(margin)이 그 이상일 때만 개입한다. margin 이 없는 판정(캐시·API)은 통과.
        m = j.get("margin")
        def _ok(kind: str) -> bool:
            thr = p.get(kind + "_margin")
            return thr is None or m is None or float(m) >= float(thr)
        if hit == 1:
            if j["v"] == 0 and "suppress" in p and rank >= _CONF_RANK[p["suppress"]] and _ok("suppress"):
                out[v] = (0, "")
            elif j["v"] == 1 and not quote and j.get("q") and v not in ABSENCE and idx.find(j["q"]):
                out[v] = (1, j["q"])                    # 규칙 근거가 비면 LLM 인용으로 보강
        else:
            if j["v"] == 1 and "recover" in p and rank >= _CONF_RANK[p["recover"]] and _ok("recover"):
                if v in ABSENCE:
                    out[v] = (1, "")
                elif j.get("q") and idx.find(j["q"]):    # 인용이 원문에 실재해야 복구
                    out[v] = (1, j["q"])
    return out


JUDGE_CACHE_VERSION = 1


def dump_judge_cache(path: str, recs: List[Dict[str, Any]], judges: List[Dict[str, Dict[str, Any]]],
                     n_done: int, n_total: int) -> None:
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with io.open(path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"_meta": {"version": JUDGE_CACHE_VERSION, "n": len(recs),
                                           "queries_done": n_done, "queries_total": n_total}},
                                ensure_ascii=False) + "\n")
            for rec, j in zip(recs, judges):
                fh.write(json.dumps({"id": rec["id"], "j": j}, ensure_ascii=False) + "\n")
        log(f"판정 캐시 기록: {path} ({len(recs)}건 · 질의 {n_done}/{n_total})")
    except Exception as e:
        log(f"  ! 판정 캐시 기록 실패: {type(e).__name__}: {e}")


def load_judge_cache(path: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    with io.open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "_meta" in d:
                continue
            out[d["id"]] = d.get("j") or {}
    return out


# ===================================================================================
# 9. 사실(JSON) 파싱
# ===================================================================================
FACT_DEFAULT: Dict[str, Any] = {
    "inst": 0, "inst_q": None, "perf": 0, "perf_amt": 0, "perf_org": 0, "perf_q": None,
    "reg": 0, "reg_lv": "없음", "reg_q": None, "size": "없음", "size_q": None,
    "dpc": 0, "dpc_q": None, "cmp": 0, "model": 0, "model_q": None,
    "pled": 0, "pled_q": None, "brief": 0, "brief_q": None,
    "sw": 0, "swlim": 0, "exc": 0, "share": -1.0, "share_q": None,
    "mism": 0, "mism_q": None, "brief_date": None, "prop_due": None,
}
_INT_KEYS = ("inst", "perf", "perf_org", "reg", "dpc", "cmp", "model", "pled",
             "brief", "sw", "swlim", "exc", "mism")
_STR_KEYS = ("inst_q", "perf_q", "reg_q", "size_q", "dpc_q", "model_q", "pled_q",
             "brief_q", "share_q", "mism_q", "brief_date", "prop_due")


def _flag(v: Any) -> int:
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return 1 if int(v) == 1 else 0
    s = str(v).strip().lower()
    return 1 if s in ("1", "true", "y", "yes", "예") else 0


def _quote(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = unicodedata.normalize("NFC", str(v)).strip()
    return s or None


def parse_facts(text: str) -> Tuple[Dict[str, Any], bool]:
    """구조화 출력 JSON → 사실 dict. 실패하면 기본값과 False."""
    obj: Any = None
    if text:
        t = text.strip()
        try:
            obj = json.loads(t)
        except Exception:
            m = re.search(r"\{[\s\S]*\}", t)
            if m:
                try:
                    obj = json.loads(m.group(0))
                except Exception:
                    obj = None
    if not isinstance(obj, dict):
        return dict(FACT_DEFAULT), False

    f = dict(FACT_DEFAULT)
    for k in _INT_KEYS:
        if k in obj:
            f[k] = _flag(obj[k])
    for k in _STR_KEYS:
        if k in obj:
            f[k] = _quote(obj[k])
    if obj.get("size") in SIZE_ENUM:
        f["size"] = obj["size"]
    if obj.get("reg_lv") in REGION_ENUM:
        f["reg_lv"] = obj["reg_lv"]
    try:
        f["perf_amt"] = max(0, int(float(obj.get("perf_amt", 0) or 0)))
    except Exception:
        f["perf_amt"] = 0
    try:
        s = float(obj.get("share", -1) if obj.get("share") is not None else -1)
        f["share"] = s if 0 < s <= 100 else -1.0
    except Exception:
        f["share"] = -1.0
    return f, True


# LLM 사실 중 결정론 규칙보다 신뢰할 수 있어 채택하는 필드.
# dev 200 + 실제 gemma facts 로 그리디 전방선택한 결과, heuristic 기반에
# 아래 실적(perf) 계열만 LLM 값으로 덮을 때 Macro F1 이 오른다
# (0.737656 → 0.745000). 나머지 LLM 필드는 과탐/노이즈로 점수를 떨어뜨린다.
LLM_TRUSTED_KEYS = ("perf", "perf_org", "perf_q", "perf_amt")


def merge_facts(llm_f: Dict[str, Any], heur_f: Dict[str, Any]) -> Dict[str, Any]:
    """결정론 heuristic 을 기반으로, 신뢰 가능한 LLM 필드만 선택적으로 덮는다.

    평가 서버는 LLM 이 GPU 로 돌지만, 실측(dev 200 + 실제 gemma) 결과 LLM 사실을
    통째로 쓰면 0.583, heuristic 만 쓰면 0.738 이다. LLM 은 실적(perf) 계열에서만
    이득을 주므로 그 필드만 채택해 0.745 를 얻는다.

    [대폭 수정 2026-09-19b] LLM perf=0이고 heuristic perf=1일 때:
    기존: LLM(=0)으로 덮어씌움 → v2/v8 대량 FN의 원인.
    변경: OR 결합 — heuristic이 탐지한 실적을 LLM이 부정하지 않도록.
    perf_org도 동일하게 OR 결합.
    """
    f = dict(heur_f)
    for k in LLM_TRUSTED_KEYS:
        if k in llm_f:
            # [변경] 이진 필드(perf, perf_org)는 OR 결합: heuristic이 1이면 유지
            if k in ("perf", "perf_org") and heur_f.get(k) == 1:
                continue  # heuristic 1 유지 (LLM이 0이어도 덮지 않음)
            f[k] = llm_f[k]
    # [PATCH13] LLM perf=1(heuristic 0)인데 인용 줄이 제출서류·평가기준 줄이면 실적 '요구'가 아니다
    #   (무라벨 300건 실측: LLM 단독 v2 양성 5건 중 4건이 "수행실적 [서식3호] 1부"·"이행실적 평가 기준" 류).
    if f.get("perf") == 1 and heur_f.get("perf") != 1:
        pq = str(f.get("perf_q") or "")
        if (not pq or not _perf_line_ok(pq) or _RE_PERF_EVAL.search(pq)
                or not re.search(r"실적|경험|경력|수행\s*이력", pq)):
            f["perf"] = 0
            f["perf_amt"] = int(heur_f.get("perf_amt") or 0)
            if heur_f.get("perf_org") != 1:
                f["perf_org"] = 0
    return f


_RE_PERF_WORD = re.compile(
    r"납품\s*실적|시공\s*실적|수행\s*실적|이행\s*실적|계약\s*실적|공급\s*실적"
    r"|운영\s*실적|관리\s*실적|유사\s*(?:용역|사업|업무)\s*(?:수행|이행|납품)?\s*실적"
    r"|실적\s*(?:금액|합계|총액|규모)"
    r"|실적\s*(?:을\s*)?(?:갖춘|보유한|가진|있는)")


def perf_requirement_line(qual: str) -> str:
    """참가자격 본문에서 '실적을 요구하는' 줄만 돌려준다(원문 부분문자열).

    - 요구 표지(이상/보유/제한 …)가 있으면 서류 어휘가 섞여 있어도 채택.
    - 요구 표지 없이 서류·평가 어휘(증명서/각 1부/평가/현황 …)만 있으면 제출목록으로 보고 건너뜀.
    """
    if not qual:
        return ""
    for m in _RE_PERF_WORD.finditer(qual):
        line = _line_of(qual, m.start(), m.end())
        if _perf_line_ok(line):
            return line[:260]
    return ""


def heuristic_facts(rec: Dict[str, Any], pre: Dict[str, Any]) -> Dict[str, Any]:
    """LLM 없이(또는 시간 초과 시) 쓰는 보수적 규칙 기반 사실."""
    f = dict(FACT_DEFAULT)
    qual = pre["qual"] or notice_text(rec)[:6000]
    nt = notice_text(rec)

    if pre.get("dpc_det"):
        f["dpc"], f["dpc_q"] = 1, pre.get("dpc_quote") or ""
    if pre.get("size_det"):
        f["size"], f["size_q"] = pre["size_det"], pre.get("size_det_q") or ""
    if pre.get("reg_det"):
        f["reg"] = 1
        f["reg_lv"] = "기초" if pre.get("reg_basic") else ("복수광역" if pre.get("reg_multi") else "광역")
        f["reg_q"] = pre.get("reg_quote") or line_with(qual, r"지역\s*제한|본점|주된\s*영업소|소재지")
    # [변경] 실적 문구는 '요구 조건'으로 쓰인 줄만 채택한다. 제출서류 목록("납품실적증명서 각 1부")·
    # 평가항목("납품실적, 견본제작 … 평가")·현황표("사업수행 실적 현황")는 실적제한이 아니어서
    # v2/v8 과탐(dev FP 5건, 비라벨 v2 2.3x·v8 3.1x)의 주원인이었다. → perf_requirement_line().
    pq = perf_requirement_line(qual)
    if pq:
        f["perf"] = 1
        f["perf_q"] = pq
        f["perf_amt"] = parse_amount_won(pq)
    if pre["comp_code_hit"]:
        f["cmp"] = 1
    # [2026-09-19c] cmp 확장(comp_joh/comp_panro9/comp_name_hit/텍스트패턴) 전부 제거.
    # comp_joh/comp_panro9 추가 시 dev v12 FN+1, v14 FN+1 (cmp 과탐 → 경쟁제품 예외 처리),
    # SL에는 영향 없음. comp_code_hit만으로 BASELINE과 동일하게 유지.
    if pre["share_hint"]:
        f["share"] = min(pre["share_hint"])
        f["share_q"] = line_with(nt, r"지분율|출자\s*비율|참여\s*비율")
    if pre["amount_mismatch"]:
        f["mism"], f["mism_q"] = 1, pre["amount_mismatch_q"]
    if pre.get("sw_det"):
        f["sw"] = 1
    # [대폭 수정 2026-09-19b] 추가 heuristic 탐지
    # inst: backstop에서 놓치는 경우 전문 추가 검색
    if f.get("inst") != 1 and pre.get("inst_det"):
        f["inst"] = 1
        f["inst_q"] = pre.get("inst_quote", "")
    # pled: 확약서 탐지 — pledge_strong(입찰 전 제출 확인된 것)만 사용
    # pledge_det(넓은 탐지)는 v19 FP 폭발 원인이므로 제외(dev FP 0→8)
    if f.get("pled") != 1 and pre.get("pledge_strong"):
        f["pled"] = 1
        f["pled_q"] = pre.get("pledge_strong_q", "")
    # brief: 설명회 참석 강제 탐지
    if f.get("brief") != 1 and pre.get("brief_gate"):
        f["brief"] = 1
        f["brief_q"] = pre.get("brief_gate_quote", "")
    # model: 모델명 탐지
    if f.get("model") != 1 and pre.get("model_det"):
        f["model"] = 1
        f["model_q"] = pre.get("model_quote", "")
    # mismatch: 불일치 탐지 강화
    if f.get("mism") != 1:
        if pre.get("method_mismatch"):
            f["mism"] = 1
            f["mism_q"] = pre.get("method_mismatch_q", "")
        elif pre.get("biz_mismatch"):
            f["mism"] = 1
            f["mism_q"] = pre.get("biz_mismatch_q", "")
    if pre.get("sw_limit"):
        f["swlim"] = 1
    if pre.get("pledge_strong"):
        f["pled"], f["pled_q"] = 1, pre.get("pledge_strong_q") or ""
    if pre.get("brief_gate"):
        f["brief"], f["brief_q"] = 1, pre.get("brief_gate_quote") or ""
    return f


# ===================================================================================
# 10. 규칙 엔진 — 사실 → v1..v24
# ===================================================================================
def line_with(text: str, pattern: str, maxlen: int = 260) -> str:
    """정규식이 걸리는 줄을 원문 그대로 돌려준다(반드시 원문 부분문자열)."""
    if not text:
        return ""
    m = re.search(pattern, text)
    if not m:
        return ""
    s = text.rfind("\n", 0, m.start()) + 1
    e = text.find("\n", m.end())
    if e < 0:
        e = len(text)
    seg = text[s:e]
    if len(seg.strip()) < 12:                       # 너무 짧으면 다음 줄까지
        e2 = text.find("\n", e + 1)
        seg = text[s: e2 if e2 > 0 else len(text)]
    return seg.strip()[:maxlen]


FALLBACK_PAT: Dict[str, str] = {
    "v1": r"참가\s*자격|참가자격",
    "v2": r"납품\s*실적|수행\s*실적|이행\s*실적|계약\s*실적|공급\s*실적|실적",
    "v3": r"납품\s*실적|수행\s*실적|이행\s*실적|계약\s*실적|공급\s*실적|실적",
    "v4": r"발주[^\n]{0,20}실적|실적[^\n]{0,20}발주|납품\s*실적|수행\s*실적|실적",
    "v5": r"지역\s*제한|본점|주된\s*영업소|소재지",
    "v6": r"지역\s*제한|본점|주된\s*영업소|소재지",
    "v7": r"지역\s*제한|본점|주된\s*영업소|소재지",
    "v8": r"지역\s*제한|본점|주된\s*영업소|소재지",
    "v9": r"제조\s*사|제조사|모델\s*명|모델명|상표|규격",
    "v12": r"직접\s*생산\s*확인",
    "v13": r"소기업|소상공인",
    "v14": r"중소기업|소기업|소상공인",
    "v15": r"소기업|소상공인",
    "v17": r"중소기업",
    "v19": r"확약서|공급\s*확약|기술\s*지원",
    "v21": r"지분율|출자\s*비율|참여\s*비율|공동\s*수급",
    "v22": r"현장\s*설명|사업\s*설명회|설명회",
    "v23": r"제안요청\s*설명|현장\s*설명|사업\s*설명회|설명회",
    "v24": r"추정\s*가\s*격|배정\s*예산|사업\s*예산|계약\s*방법",
}


def _ymd(s: Optional[str]) -> Optional[Tuple[int, int, int]]:
    if not s:
        return None
    d = parse_dates(str(s))
    return d[0] if d else None


def required_brief_days(est: int) -> int:
    for lo, days in NEGO_BRIEF_DAYS:
        if est >= lo:
            return days
    return 10


_WIDE_REGIONS = {"서울": "서울", "부산": "부산", "대구": "대구", "인천": "인천", "광주": "광주", "대전": "대전", "울산": "울산",
                 "세종": "세종", "경기": "경기", "강원": "강원", "충북": "충북", "충청북도": "충북", "충남": "충남", "충청남도": "충남",
                 "전북": "전북", "전라북도": "전북", "전남": "전남", "전라남도": "전남", "경북": "경북", "경상북도": "경북",
                 "경남": "경남", "경상남도": "경남", "제주": "제주"}
_RE_WIDE_REGION = re.compile(r"(?<![가-힣])(?:서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충청북도|충북|충청남도|충남|전라북도|전북|전라남도|전남|경상북도|경북|경상남도|경남|제주)(?=특별|광역|자치|도|시|[^가-힣]|$)")


def _region_or_mismatch(rec) -> Tuple[bool, str]:
    """공고문 소재지 요건 줄이 메타 제한지역에 없는 광역지역을 함께 허용하면 (True, 줄)."""
    m = rec.get("meta") or {}
    if str(m.get("지역제한여부") or "") != "Y":
        return False, ""
    meta_reg = str(m.get("제한지역코드목록") or "")
    meta_set = {_WIDE_REGIONS[x] for x in _RE_WIDE_REGION.findall(meta_reg)}
    if not meta_set:
        return False, ""
    for ln in notice_text(rec).split("\n"):
        if "소재" not in ln or not re.search(r"또는|및|,", ln):
            continue
        if not re.search(r"(?:소재지|사업장|본점)[^\n]{0,120}(?:둔|두고|소재한|소재하고|소재하는|있는)\s*(?:자|업체|기업|법인|사업자)", ln):
            continue
        found = {_WIDE_REGIONS[x] for x in _RE_WIDE_REGION.findall(ln)}
        if len(found) >= 2 and found - meta_set:
            return True, ln.strip()
    return False, ""


def apply_rules(f: Dict[str, Any], rec: Dict[str, Any], pre: Dict[str, Any]) -> Dict[str, Tuple[int, str]]:
    """법령 임계값·상호배타·부재탐지를 파이썬에서 결정한다. 값 = (위반여부, LLM 인용문)."""
    est = pre["est"]
    local = pre["local"]
    band = pre["band"]
    budget = pre["budget"] or int(est * 1.1)
    thr = region_threshold(local)
    smallq = is_small_quote(rec)
    nego = is_negotiation(rec)
    local_smallq = local and smallq                       # 지방+소액수의: v2·v6·v7·v8 예외
    # 수의계약(경쟁입찰 아님): v11·v13·v14~v18 제외. 메타가 경쟁입찰로 잘못 입력된 건은 공고문 머리말로 보완.
    sole = smallq or ("수의" in meta_str(rec, "계약방법")) or is_sole_notice(rec)

    r: Dict[str, Tuple[int, str]] = {v: (0, "") for v in ITEMS}

    def put(v: str, hit: bool, q: Optional[str]) -> None:
        r[v] = (1 if hit else 0, q or "")

    # --- 참가자격: 기관·실적 -------------------------------------------------
    # 결정론 백스톱 OR LLM (v5~v7 과 동일한 앙상블 구조로 맞춘다)
    put("v1", f["inst"] == 1 or pre.get("inst_det", False),
        f["inst_q"] or pre.get("inst_quote") or "")

    # 요구 실적금액이 적힌 줄 자체가 실적제한의 직접 증거다(정규식보다 재현율이 높다).
    perf_amt = max(int(f["perf_amt"] or 0), int(pre.get("perf_amt_det", 0) or 0))
    perf = f["perf"] == 1 or pre.get("perf_det", False) or perf_amt > 0
    perf_q = f["perf_q"] or pre.get("perf_quote") or pre.get("perf_amt_q") or ""
    put("v2", perf and est < GOSI_NATIONAL and not local_smallq, perf_q)
    # [변경] 국가계약법 시행규칙 제25조·지방 집행기준은 실적 금액제한을 "추정가격의 1배 이내"로 허용하고
    #        "1배를 초과"하는 제한을 위법 사례로 든다 → 요구 실적 == 예산(1배)은 위반이 아니다(초과만 위반).
    #        자가라벨 대조 v3 FP 20건 중 12건이 '요구실적 = 기초금액/배정예산' 동액 사례였다. dev 양성 8건은 모두 초과.
    put("v3", perf_amt > 0 and budget > 0 and perf_amt > budget,
        f["perf_q"] or pre.get("perf_amt_q") or perf_q)
    # 결정론 검출은 그 자체가 "실적 + 기관"을 동시에 요구하므로 perf 게이트를 다시 걸지 않는다.
    put("v4", (perf and f["perf_org"] == 1) or pre.get("perf_org_det", False),
        f["perf_q"] or pre.get("perf_org_quote") or "")

    # --- 참가자격: 지역제한 --------------------------------------------------
    # 문서 전문 + 나라장터 제한지역코드목록 기반의 결정적 검출기를 1순위로 쓰고,
    # LLM 판단은 검출기가 놓친 경우를 보완하는 용도로만 OR 결합한다.
    reg = bool(pre.get("reg_det")) or f["reg"] == 1 or f["reg_lv"] in ("광역", "기초", "복수광역")
    basic = bool(pre.get("reg_basic")) or f["reg_lv"] == "기초"
    multi = bool(pre.get("reg_multi")) or f["reg_lv"] == "복수광역"
    reg_q = f["reg_q"] or pre.get("reg_quote") or ""
    put("v5", reg and est >= thr, reg_q)
    # [변경] 소액수의 견적은 국가계약법 시행령 제30조 체계에서도 시·군·구 단위 제한이 관행적으로 허용된다 →
    #        지방 여부와 무관하게 예외(무라벨 20k v6 FP 28건 중 17건이 국가법+소액수의; dev 양성 6건은 모두 비소액수의).
    put("v6", reg and basic and est < thr and not smallq, reg_q)
    put("v7", reg and multi and est < thr and not local_smallq, reg_q)
    put("v8", reg and perf and not local_smallq, reg_q or perf_q)

    # --- 규격서 특정 모델 ----------------------------------------------------
    put("v9", f["model"] == 1 or pre.get("model_det", False),
        f["model_q"] or pre.get("model_quote") or "")

    # --- 중기간 경쟁제품 / 일반제품 분기 -------------------------------------
    # 판로지원법 제9조(직접생산확인) 인용은 용역 부문 경쟁제품 공고의 강한 표지다.
    # 분기 재현율 9/15 → 11/15, 오발화 3 → 5 (분기 F1 0.667 → 0.710).
    cmp_hit = (f["cmp"] == 1 or pre["comp_code_hit"] or pre.get("comp_name_hit", False)
               or pre.get("comp_joh", False) or pre.get("comp_panro9", False)
               or pre.get("comp_text_hit", False) or pre.get("comp_event", False)
               or pre.get("comp_sysmaint", False))
    # [추가] 명시된 세부품명번호가 고시 목록에 있으나 특이사항의 추정가격 조건을 벗어난 경우
    #        (dev 054: 축제기획및대행서비스 '3억원 미만에 한함' vs 추정 3.27억) 이 건은 경쟁제품이 아니다.
    #        조항호·판로지원법 제9조 인용만으로 분기를 유지하면 v12·v14 를 놓친다 → 일반제품 분기로 보낸다.
    if pre.get("comp_excluded") and not pre["comp_code_hit"] and f["cmp"] != 1:
        cmp_hit = False
    size = f["size"]
    if size in ("없음", "기타") and pre.get("size_det"):
        size = pre["size_det"]                     # LLM 이 비워둔 자리만 결정론으로 메운다
    no_size = size in ("없음", "기타")
    exc = f["exc"] == 1 or pre.get("exception_doc", False) or pre.get("exc_det", False) or pre.get("exc_joh", False)
    dpc = f["dpc"] == 1 or pre.get("dpc_det", False)

    # 분기 자체는 넓게 잡아 v12·v14~v18 을 억제하되, 경쟁제품 위반 '주장'은
    # 근거가 강한 신호(세부품명번호 일치 또는 LLM 확정)에서만 낸다.
    # v13 은 '경쟁제품이라는 사실'이 확정적일 때만 주장한다. 세부품명번호 일치·
    # LLM 확정·판로지원법 제9조 인용은 확정 신호, 품명 퍼지매칭은 아니다.
    # [조사기록 2026-09-10] 남은 v13 FP 2건(dev 056·193)은 규칙을 더 조여서 없앨 수 없다 —
    #   · 056: 자격문에 '직접생산확인증명서(세부품명번호 8111219901 인터넷지원개발서비스)' 명시 + 소기업 제한.
    #          경쟁제품이 확정적인데 정답은 v12=1(=경쟁제품 아님 전제) → 정답끼리 모순.
    #   · 193: 8014199001(기타행사기획및대행서비스)·est 5천만·소기업 제한 → dev 074(같은 코드·est 6.4천만·
    #          같은 제한)는 정답 v13=1 인데 193 은 0. 텍스트로 분리 가능한 차이가 없다.
    #   판로지원법 제7조의2제1항이 '중기부장관이 지정한 경쟁제품은 소기업·소상공인 제한경쟁 가능'을 열어 두어
    #   그 지정목록이 있어야 갈리는데 법령패키지에 없다 → 규칙 강제 대신 판정(JUDGE_POLICY v13 suppress)에 맡긴다.
    cmp_strict = (f["cmp"] == 1 or pre["comp_code_hit"]
                  or pre.get("comp_panro9", False) or pre.get("comp_text_hit", False))
    # 품명 퍼지매칭만으로 발화한 분기는 근거가 약하다 — 부재탐지 주장에서 제외한다.
    cmp_named = bool(f["cmp"] == 1 or pre["comp_code_hit"]
                     or pre.get("comp_joh", False) or pre.get("comp_panro9", False)
                     or pre.get("comp_text_hit", False) or pre.get("comp_event", False)
                     or pre.get("comp_sysmaint", False))
    svc = not pre.get("is_goods", False)          # 용역 공고에서만 v10/v11/v13 을 주장한다
    if cmp_hit:
        exc_cmp = bool(pre.get("exception_doc", False) or pre.get("exc_det", False))   # 조항호 예외(exc_joh)는 제외(dev 062)
        # v10 은 문서형 예외만 인정(dev 069: 시행령 제2조의3제2호 '비영리법인 확인서 면제' 인용은 직생확인 예외가 아니다)
        put("v10", svc and cmp_named and not dpc and not pre.get("dpc_any", False)
            and not pre.get("exception_doc", False), None)                            # 부재탐지
        # [변경] v13 과 같은 논리 — 판로지원법 제7조의 규모 제한 의무는 '중소기업자간 경쟁입찰' 전제이므로
        #        수의계약(소액수의 견적)에는 성립하지 않는다(무라벨 20k v11 FP 31건 중 15건이 수의계약). 예외조항은 전문에서 인정.
        put("v11", svc and no_size and not exc_cmp and not sole, None)               # 부재탐지
        # 수의계약(소액수의 견적)은 판로지원법 제7조의 '중소기업자간 경쟁입찰' 전제가 없다 → v13 미주장
        # (dev 양성 6/6 모두 경쟁입찰 · dev FP 107 수의 · 무라벨 20k 발화 5.6% 중 다수가 소액수의 견적)
        put("v13", svc and cmp_strict and size == "소기업소상공인" and not sole, f["size_q"])
    else:
        put("v12", dpc, f["dpc_q"] or pre.get("dpc_quote") or "")
        if band == "over_gosi":
            # [변경] 규모 '과잉제한' 항목도 수의계약에는 성립하지 않는다 — 판로지원법 시행령 제2조의2 의
            #        규모별 의무는 '제한경쟁입찰' 방법을 정한 것이고, 제2조의3제1항제3호는 수의계약을 우선조달
            #        예외로 명시한다. 근거 없는 제한이 아니라 애초에 조문이 적용되지 않는 국면이다(dev 144).
            # [PATCH13d] 소프트웨어사업(사업금액 20억 미만)은 소프트웨어 진흥법 제48조에 따라 대기업 참여가 제한되어
            #        중소기업 제한이 법령 근거를 가진다(자가라벨 v14 FP 007321·020046) → 중소기업 전체 제한은 v14 아님.
            sw_small = bool(pre.get("sw_det")) and svc and 0 < int(est * 1.1) < 2_000_000_000
            put("v14", size == "소기업소상공인" or (size == "중소기업" and not sw_small), f["size_q"])
        elif band == "mid":                                   # 1억 이상 ~ 고시금액 미만
            put("v15", size == "소기업소상공인", f["size_q"])
            # [변경·§10-4] 부재탐지는 '자격 조항을 실제로 읽었을 때'만 성립한다.
            #   qual_found : 서약서·부정당업자 안내만 있고 자격 조항이 없는 공고에서 주장 금지(dev200 FP 149)
            #   size_doc_hint : 중소기업·소상공인 확인서 제출을 요구하면 규모 제한이 사실상 존재(dev200 FP 132)
            put("v16", no_size and not exc and not sole
                and pre.get("qual_found", True), None)
        else:                                                 # 1억원 미만
            # [변경] 추정가격이 자리표시자(1원·5원 등, dev 094·142·150 다품목 의료기기 공고)면 금액대를 알 수 없으므로
            #        '1억 미만 중소기업 제한' 주장을 하지 않는다(정답 모두 0). v18 의 est > 0 게이트와 같은 취지.
            # [변경·Run04] 공고 본문에 "중소기업자간 경쟁제품" 이 명시된 경우는 경쟁제품 예외(③)에 해당 →
            #   comp_code_hit 로 안 잡힌 품목코드(예: 아스팔트 3012160101)도 본문 명시로 검출(dev FP -1).
            _cmp_text_v17 = bool(re.search(r"중소기업자간\s*경쟁제품", notice_text(rec)))
            put("v17", size == "중소기업" and est >= 1000 and not _cmp_text_v17, f["size_q"])
            # [변경] v18 부재탐지는 경쟁입찰에만 적용. 판로지원법 시행령 제2조의2 의 소기업·소상공인
            # 제한 의무는 '경쟁입찰' 전제이므로 수의계약(소액수의 견적 포함)에는 성립하지 않는다.
            # [변경 전] 수의계약도 발화 → dev FP 3건(097·172·195), 비라벨 500건 4.46x 과발화
            # [변경 후] 수의계약 제외(dev 양성 7건은 모두 제한경쟁 → 재현율 손실 없음)
            # 추정가격·예산이 모두 비어 있으면(비라벨 10건) 금액대를 알 수 없으므로 주장하지 않는다.
            # [변경] v18 은 qual_found 만 적용. size_doc_hint(확인서 제출요구)까지 걸면
            #        dev200 FN 이 1→4 로 늘었다 — 1억 미만 구간의 의무는 '소기업·소상공인' 제한인데
            #        '중소기업 확인서' 요구는 그 의무 이행의 증거가 아니기 때문(v16 구간과 다르다).
            put("v18", no_size and not exc and not sole and est > 0
                and pre.get("qual_found", True), None)

    # --- 확약서 --------------------------------------------------------------
    # 항목 정의가 "물품공급 확약서"이므로 물품 공고에만 성립한다(dev 양성 6/6 물품(내자)).
    # [변경] 결정론 강검출(pledge_strong)도 물품 공고 게이트를 공유한다. 20k 규칙 양성 42건 중 7건이 일반용역
    #        (정보시스템·계측기 유지관리)이었고 자가라벨(dev 대조 P/R 1.00 항목)은 7건 모두 '물품 공고 아님'으로 판정.
    # [변경] 자가라벨 대조 v19 FP 6/6 이 '계약 시 제출'·'적격심사시 제출' 문맥: 입찰 전 제출이 아니므로 위반 아님.
    #        pledge_strong 은 이미 입찰 전 시점을 검증했으므로 LLM pled=1 경로만 필터링한다.
    _pled_hit_19 = f["pled"] == 1 or pre.get("pledge_strong", False)
    if _pled_hit_19 and not pre.get("pledge_strong", False):
        _pq19 = (f["pled_q"] or "")
        if re.search(r"계약\s*(?:체결\s*)?시|적격심사시|적격심사\s*서류|낙찰(?:예정)?자", _pq19):
            _pled_hit_19 = False
    put("v19", _pled_hit_19 and pre.get("is_goods", True),
        f["pled_q"] or pre.get("pledge_strong_q") or "")

    # --- 소프트웨어사업 대기업 참여제한 명시 여부(부재탐지) -------------------
    sw_hit = bool(pre.get("sw_det")) or f["sw"] == 1
    sw_lim = bool(pre.get("sw_limit")) or f["swlim"] == 1
    # [변경] 물품 공고는 건명이 SW 성격일 때만 주장(무라벨 20k v20 FP 19건 중 14건이 하드웨어 물품 구매).
    sw_scope = (not pre.get("is_goods", False)) or bool(_RE_SW_GOODS_TITLE.search(
        guess_title(notice_text(rec)) + " " + meta_str(rec, "세부품명목록") + " " + meta_str(rec, "입찰건명")))
    put("v20", sw_hit and not sw_lim and sw_scope, None)

    # --- 공동수급 최소지분율 --------------------------------------------------
    share = min(pre["share_hint"]) if pre["share_hint"] else (f["share"] if f["share"] > 0 else -1.0)
    min_share = MIN_SHARE_LOCAL if local else MIN_SHARE_NATIONAL
    put("v21", 0 < share < min_share, f["share_q"])

    # --- 협상에 의한 계약: 설명회 참석 강제(v22) ------------------------------
    brief_gate = bool(pre.get("brief_gate")) or f["brief"] == 1
    brief_q = f["brief_q"] or pre.get("brief_gate_quote") or ""
    put("v22", nego and brief_gate, brief_q)

    # --- 협상(지방)에 의한 계약: 설명회~제안서 제출마감 기간(v23) --------------
    v23 = False
    if local and nego:
        bd = pre.get("brief_day") or _ymd(f["brief_date"])
        pd = pre.get("prop_day") or _ymd(f["prop_due"])
        if bd and pd:
            gap = days_between(bd, pd)
            # [2026-09-19d] gap<=0(설명회가 마감 후이거나 동일일)도 기간 부족으로 위반.
            # 기존: 0 < gap 만 허용 → SL v23 FN 중 gap<=0이 8건.
            if gap < required_brief_days(est):
                v23 = True
    put("v23", v23, brief_q)

    # --- 공고문 vs 나라장터 입력값 --------------------------------------------
    # LLM 의 mism 판단은 과탐이 심하다(로컬 프록시 LLM 실측: 8건 중 6건이 정답 0인데 1로 판정).
    # 계약방법·지역·업종 축의 결정적 대조도 dev 200건에서 F1 0.063 수준으로 무의미했고,
    # 금액·계약방법·업종코드 세 축의 결정적 대조를 OR 로 묶는다(dev F1 0.286 → 0.421).
    # 지역 축은 검출기 오차 때문에 FP 가 급증해 제외했다.
    # [PATCH13e] 지역 축(좁은 형태): 메타 지역제한=Y 인데 공고문 소재지 요건 줄이 메타에 없는 광역지역을 '또는'으로
    #        추가 허용한 경우(dev 072: 공고 '경기도 또는 제주도' vs 메타 '경기도'). 광역명 2개 이상이 한 줄에 있을 때만 본다.
    reg_mm, reg_mm_q = _region_or_mismatch(rec)
    mm_hit = (pre["amount_mismatch"] or pre.get("method_mismatch", False) or reg_mm
              or pre.get("biz_mismatch", False)
              or pre.get("title_price_mismatch", False)
              or f["mism"] == 1)
    mm_q = (pre["amount_mismatch_q"] or pre.get("method_mismatch_q") or reg_mm_q
            or pre.get("biz_mismatch_q") or pre.get("title_price_mismatch_q")
            or (f["mism_q"] if f["mism"] == 1 else None))
    put("v24", mm_hit, mm_q)
    return r


# ===================================================================================
# 11. 근거문구 정합화 (반드시 원문 부분문자열)
# ===================================================================================
def _compact(s: str) -> Tuple[str, List[int]]:
    """공백을 제거한 문자열과 원문 인덱스 매핑."""
    chars: List[str] = []
    idx: List[int] = []
    for i, ch in enumerate(s):
        if ch.isspace():
            continue
        chars.append(ch)
        idx.append(i)
    return "".join(chars), idx


class EvidenceIndex:
    """공고 원문(문서 단위)에서 인용문을 복원한다."""

    def __init__(self, rec: Dict[str, Any]):
        self.docs: List[Tuple[str, str, List[int]]] = []
        for d in rec.get("docs", []):
            t = d.get("text") or ""
            if not t.strip():
                continue
            cm, im = _compact(t)
            self.docs.append((t, cm, im))
        if not self.docs:
            t = full_text(rec)
            cm, im = _compact(t)
            self.docs.append((t, cm, im))

    def _slice(self, doc: int, c_start: int, c_len: int) -> str:
        text, _, im = self.docs[doc]
        a = im[c_start]
        b = im[c_start + c_len - 1] + 1
        return text[a:b]

    def find(self, quote: Optional[str]) -> str:
        if not quote:
            return ""
        q = unicodedata.normalize("NFC", str(quote)).strip()
        if len(q) < 6:
            return ""
        for text, _, _ in self.docs:                       # ① 정확 일치
            if q in text:
                return q
        cq, _ = _compact(q)                                # ② 공백 무시 일치
        if len(cq) < 5:
            return ""
        for i, (_, cm, _) in enumerate(self.docs):
            p = cm.find(cq)
            if p >= 0:
                return self._slice(i, p, len(cq))
        for i, (_, cm, _) in enumerate(self.docs):         # ③ 최장 접두 일치
            lo, hi, best = 8, len(cq), 0
            while lo <= hi:
                mid = (lo + hi) // 2
                if cm.find(cq[:mid]) >= 0:
                    best, lo = mid, mid + 1
                else:
                    hi = mid - 1
            if best >= max(12, int(len(cq) * 0.45)):
                p = cm.find(cq[:best])
                return self._slice(i, p, best)
        return ""


def sanitize_evidence(ev: str) -> str:
    """부분문자열 성질을 깨지 않는 정리: 줄 선택 → 좌우 공백 제거 → 수식 접두 제거 → 500자 절단."""
    if not ev:
        return ""
    ev = unicodedata.normalize("NFC", ev)
    if "\n" in ev or "\r" in ev:
        segs = [s for s in re.split(r"[\r\n]+", ev) if s.strip()]
        if segs:
            ev = max(segs, key=lambda s: len(s.strip()))
    ev = ev.strip()
    while ev and (ev[0] in "=+@\t" or ev[0].isspace()):
        ev = ev[1:].strip()
    ev = ev[:EVIDENCE_MAX].strip()
    return ev if len(ev) >= 6 else ""


def resolve_evidence(results: Dict[str, Tuple[int, str]], rec: Dict[str, Any],
                     idx: EvidenceIndex) -> Tuple[Dict[str, Dict[str, Any]], int, int]:
    """위반=1인 항목만 근거를 남기고, 원문 부분문자열로 강제한다."""
    src = full_text(rec)
    qual = extract_qualification(src) or src
    kept = dropped = 0
    out: Dict[str, Dict[str, Any]] = {}
    for v in ITEMS:
        hit, quote = results.get(v, (0, ""))
        ev = ""
        if hit == 1 and v not in ABSENCE:
            ev = sanitize_evidence(idx.find(quote))
            if not ev and quote:
                dropped += 1
            if not ev:                                     # LLM 인용 실패 → 규칙 기반 대체
                pat = FALLBACK_PAT.get(v)
                if pat:
                    ev = sanitize_evidence(line_with(qual, pat) or line_with(src, pat))
                    if ev and not idx.find(ev):
                        ev = ""
            if ev:
                kept += 1
        out[v] = {"위반여부": int(hit), "근거문구": ev}
    return out, kept, dropped


# ===================================================================================
# 12. submission.csv 저장 · 자가검증
# ===================================================================================
def to_row(rec_id: str, judgment: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    row: Dict[str, Any] = {"id": rec_id}
    for i, v in enumerate(ITEMS, 1):
        row[v] = judgment[v]["위반여부"]
        row[f"e{i}"] = judgment[v]["근거문구"]
    return row


def empty_row(rec_id: str) -> Dict[str, Any]:
    return to_row(rec_id, {v: {"위반여부": 0, "근거문구": ""} for v in ITEMS})


def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with io.open(tmp, "w", encoding="utf-8", newline="") as f:   # UTF-8(BOM 없음) · RFC4180
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: unicodedata.normalize("NFC", str(r[k])) for k in COLUMNS})
    os.replace(tmp, path)


def validate_csv(path: str, expected_ids: List[str]) -> List[str]:
    """열 49 · 행 수 · id 유일/일치 · v 0/1 · e 500자 이하 · 부재탐지 공란 · 수식 접두 없음"""
    errs: List[str] = []
    with io.open(path, "r", encoding="utf-8", newline="") as f:
        rd = csv.reader(f)
        header = next(rd, None)
        rows = list(rd)
    if header != COLUMNS:
        return [f"헤더 불일치: {len(header or [])}열 (기대 {len(COLUMNS)})"]
    if len(rows) != len(expected_ids):
        errs.append(f"행 수 {len(rows)} ≠ 입력 {len(expected_ids)}")
    ids = [r[0] for r in rows]
    if len(set(ids)) != len(ids):
        errs.append("id 중복")
    if set(ids) != set(expected_ids):
        errs.append(f"id 집합 불일치 (누락 {len(set(expected_ids) - set(ids))})")
    absence_idx = {COLUMNS.index("e" + v[1:]) for v in ABSENCE}
    for r in rows:
        if len(r) != len(COLUMNS):
            errs.append(f"{r[0]}: 열 수 {len(r)}")
            continue
        if any(x not in ("0", "1") for x in r[1:25]):
            errs.append(f"{r[0]}: 위반여부에 0/1 아닌 값")
        if any(len(x) > EVIDENCE_MAX for x in r[25:]):
            errs.append(f"{r[0]}: 근거문구 {EVIDENCE_MAX}자 초과")
        if any(r[j] for j in absence_idx):
            errs.append(f"{r[0]}: 부재탐지 항목에 근거문구")
        if any(x.startswith(("=", "+", "@")) for x in r[25:]):
            errs.append(f"{r[0]}: 수식 접두 근거문구")
    return errs[:20]


# ===================================================================================
# 12-b. 사실 캐시 — LLM 추출과 규칙 적용의 분리
# ===================================================================================
# GPU 가 있는 곳에서 LLM 사실추출을 1회 돌려 캐시해 두면, 규칙 엔진 변경은 GPU 없이
# 그 캐시를 재생하여 정확히 평가할 수 있다. 규칙은 facts 만 소비하기 때문이다.
# 캐시는 개발·검증 전용이며 제출 실행 경로에는 관여하지 않는다.
FACT_CACHE_VERSION = 1


def dump_fact_cache(path: str, recs: List[Dict[str, Any]],
                    facts: List[Dict[str, Any]], llm_done: int, done: int) -> None:
    """레코드 id → LLM 이 채운 facts 를 JSONL 로 남긴다."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with io.open(path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"_meta": {"version": FACT_CACHE_VERSION,
                                           "n": len(recs), "llm_done": llm_done,
                                           "llm_attempted": done}},
                                ensure_ascii=False) + "\n")
            for rec, f in zip(recs, facts):
                fh.write(json.dumps({"id": rec["id"], "f": f}, ensure_ascii=False) + "\n")
        log(f"사실 캐시 기록: {path} ({len(recs)}건, LLM성공 {llm_done})")
    except Exception as e:                       # 캐시 실패가 제출을 막아선 안 된다
        log(f"  ! 사실 캐시 기록 실패: {type(e).__name__}: {e}")


def load_fact_cache(path: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """dump_fact_cache 가 남긴 파일을 되읽는다. (id→facts, meta)"""
    out: Dict[str, Dict[str, Any]] = {}
    meta: Dict[str, Any] = {}
    with io.open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "_meta" in d:
                meta = d["_meta"]
                continue
            out[d["id"]] = d["f"]
    if meta.get("version") != FACT_CACHE_VERSION:
        log(f"  ! 사실 캐시 버전 불일치 (파일 {meta.get('version')} vs 코드 {FACT_CACHE_VERSION})")
    return out, meta


# ===================================================================================
# 13. 실행 — 데드라인 컨트롤러 포함
# ===================================================================================
def build_rows(recs: List[Dict[str, Any]], pres: List[Dict[str, Any]],
               facts: List[Dict[str, Any]], indices: List[EvidenceIndex],
               judges: Optional[List[Dict[str, Dict[str, Any]]]] = None,
               policy: Optional[Dict[str, Dict[str, Any]]] = None) -> Tuple[List[Dict[str, Any]], int, int]:
    rows, kept, dropped = [], 0, 0
    for i, (rec, pre, f, idx) in enumerate(zip(recs, pres, facts, indices)):
        try:
            res = apply_rules(f, rec, pre)
            if judges and judges[i]:
                res = apply_judgments(res, judges[i], idx, policy)
            judgment, k, d = resolve_evidence(res, rec, idx)
            rows.append(to_row(rec["id"], judgment))
            kept += k
            dropped += d
        except Exception as e:
            log(f"  ! {rec['id']} 후처리 실패 → 전항목 0: {type(e).__name__}: {e}")
            rows.append(empty_row(rec["id"]))
    return rows, kept, dropped


# ===================================================================================
# 9-2. 추정 LB Macro (보조 채택 지표)  [추가 2026-09-10]
# ===================================================================================
# dev200 은 항목당 양성이 5~8건이 되도록 의도적으로 균형 표집됐다(양성률 ~3.2%).
# 실제 test 는 그렇지 않아 label shift 가 있고, 저빈도 항목일수록 FP 1건이 정밀도를
# 크게 깎는다(PML 5.1.4.4). 그래서 dev Macro 만 보면 채택 판단이 뒤집힌다.
#
# 여기서는 제공된 무라벨 학습셋(train_unlabeled)에 같은 규칙을 돌려 얻은 항목별 발화율을
# test 양성률의 대리값으로 삼아, dev 에서 측정한 재현율·오탐률을 그 분포로 옮겨 F1 을 다시 계산한다.
#   shift_v = (무라벨 발화율_v) / (dev 발화율_v)
#   p_test  = dev 양성률_v × shift_v
#   TP' = p·r,  FP' = (1-p)·q,  FN' = p·(1-r)     (r = dev 재현율, q = dev 음성당 오탐률)
# 규칙 준수: 대회가 제공한 무라벨 데이터의 통계만 쓴다(외부 데이터·정답 라벨 미사용).
#
# TEST_SHIFT 값 출처: train_unlabeled.jsonl 20,000건 중 6건마다 1건씩 뽑은 3,334건에
# 현재 규칙(data-dir 보정 반영)을 mock 으로 돌려 얻은 발화율 ÷ dev200 규칙 발화율.
TEST_SHIFT: Dict[str, float] = {          # 측정값(2026-09-10): 무라벨 3,334건 발화율 ÷ dev200 발화율
    "v1": 0.048, "v2": 0.891, "v3": 0.082, "v4": 0.510,
    "v5": 0.051, "v6": 0.120, "v7": 0.682, "v8": 0.540,
    "v9": 0.060, "v10": 0.247, "v11": 0.470, "v12": 0.060,
    "v13": 0.197, "v14": 0.313, "v15": 0.160, "v16": 0.600,
    "v17": 0.540, "v18": 1.207, "v19": 0.084, "v20": 0.650,
    "v21": 0.020, "v22": 0.180, "v23": 0.050, "v24": 0.345,
}
_TEST_SHIFT_RAW: Dict[str, Tuple[int, int]] = {}   # v -> (무라벨 발화건수, 무라벨 표본수)


def estimate_lb_macro(stat: Dict[str, Tuple[int, int, int]], n_dev: int = 200,
                      shift: Optional[Dict[str, float]] = None,
                      ) -> Tuple[float, Dict[str, Dict[str, float]]]:
    """항목별 (TP, FP, FN) → (추정 LB Macro F1, 항목별 상세).

    stat: {"v1": (tp, fp, fn), ...}  — dev200 기준 실측값
    shift: 항목별 test/dev 양성률 비. 없으면 TEST_SHIFT, 거기에도 없으면 1.0(=이동 없음).
    """
    sh = shift if shift is not None else TEST_SHIFT
    detail: Dict[str, Dict[str, float]] = {}
    f1s: List[float] = []
    for v in ITEMS:
        tp, fp, fn = stat.get(v, (0, 0, 0))
        pos = tp + fn
        neg = max(1, n_dev - pos)
        r = (tp / pos) if pos else 0.0                     # dev 재현율
        q = fp / neg                                       # dev 음성 1건당 오탐 확률
        k = float(sh.get(v, 1.0))
        p = min(1.0, max(0.0, (pos / n_dev) * k))          # 추정 test 양성률
        tp_e, fp_e, fn_e = p * r, (1.0 - p) * q, p * (1.0 - r)
        den = 2 * tp_e + fp_e + fn_e
        f1 = (2 * tp_e / den) if den > 0 else 0.0
        prec = (tp_e / (tp_e + fp_e)) if (tp_e + fp_e) > 0 else 0.0
        f1s.append(f1)
        detail[v] = {"dev_f1": (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0,
                     "shift": round(k, 3), "p_test": round(p, 5),
                     "prec_est": round(prec, 4), "f1_est": round(f1, 4)}
    return (sum(f1s) / len(f1s) if f1s else 0.0), detail


def stat_from_csv(pred_path: str, label_path: str) -> Tuple[Dict[str, Tuple[int, int, int]], int]:
    """제출 CSV × 정답 CSV → 항목별 (TP, FP, FN) 과 레코드 수."""
    def read(path: str) -> Dict[str, Dict[str, str]]:
        with io.open(path, encoding="utf-8-sig", newline="") as fh:
            return {r["id"]: r for r in csv.DictReader(fh)}
    P, L = read(pred_path), read(label_path)
    ids = [i for i in L if i in P]
    stat = {}
    for v in ITEMS:
        tp = fp = fn = 0
        for i in ids:
            a, b = int(P[i].get(v, 0) or 0), int(L[i].get(v, 0) or 0)
            tp += a & b
            fp += a & (1 - b)
            fn += (1 - a) & b
        stat[v] = (tp, fp, fn)
    return stat, len(ids)


def load_test_shift(unl_pred_path: str, dev_pred_path: str) -> Dict[str, float]:
    """무라벨 예측 CSV·dev 예측 CSV 의 항목별 발화율 비 → TEST_SHIFT 를 갱신한다."""
    def rate(path: str) -> Tuple[Dict[str, float], int]:
        with io.open(path, encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        n = max(1, len(rows))
        return {v: sum(int(r.get(v, 0) or 0) for r in rows) / n for v in ITEMS}, n
    ru, nu = rate(unl_pred_path)
    rd, nd = rate(dev_pred_path)
    out = {}
    for v in ITEMS:
        if rd[v] > 0:
            out[v] = ru[v] / rd[v]
        elif ru[v] > 0:
            out[v] = 1.0
        else:                                              # 양쪽 다 0발화 → 판단 보류
            out[v] = 1.0
        _TEST_SHIFT_RAW[v] = (int(round(ru[v] * nu)), nu)
    TEST_SHIFT.update(out)
    return out


def run(input_path: str, out_path: str, runner_cls, data_dir: str,
        limit: Optional[int], chunk: int, deadline: float, t_start: float,
        dump_facts: str = "", fact_cache: str = "",
        dump_judge: str = "", judge_cache: str = "", no_judge: bool = False,
        **runner_kw) -> Dict[str, Any]:
    t_all = time.time()
    recs = list(iter_records(input_path, limit=limit))
    log(f"입력 {len(recs)}건 ← {input_path}")
    if not recs:
        write_csv([], out_path)
        return {"건수": 0, "자가검증": "PASS"}

    comp_codes, comp_names = load_competition_products(data_dir)
    log(f"경쟁제품 사전: 세부품명번호 {len(comp_codes):,}개 · 품명 {len(comp_names):,}개")

    # ── ① 결정적 전처리 + 규칙 기반 임시 답안(안전망) ────────────────────────
    pres, indices, facts = [], [], []
    for rec in recs:
        try:
            pre = precompute(rec, comp_codes, comp_names)
        except Exception as e:
            log(f"  ! {rec['id']} precompute 실패: {type(e).__name__}: {e}")
            pre = {"exception_doc": False, "est": 0, "budget": 0, "local": is_local_law(rec),
                   "band": "under1e", "qual": "", "qual_found": False, "size_doc_hint": False,
                   "share_hint": [], "comp_codes": [],
                   "comp_code_hit": False, "comp_name_hit": False, "comp_text_hit": False,
                   "comp_names": [], "amount_mismatch": False,
                   "amount_mismatch_q": "", "title": ""}
        pres.append(pre)
        indices.append(EvidenceIndex(rec))
        try:
            facts.append(heuristic_facts(rec, pre))
        except Exception:
            facts.append(dict(FACT_DEFAULT))

    rows, _, _ = build_rows(recs, pres, facts, indices)
    write_csv(rows, out_path)                     # 이후 어떤 사고가 나도 제출 파일은 존재
    log(f"규칙 기반 임시 답안 기록 완료 ({time.time() - t_all:.0f}s)")

    # ── ①-b 사실 캐시 재생 ───────────────────────────────────────────────
    # GPU 에서 뽑아 둔 LLM 사실을 그대로 주입한다. 규칙 엔진만 바뀐 경우
    # 이 경로의 결과는 실제 LLM 실행과 동일하다(규칙은 facts 만 소비하므로).
    if fact_cache:
        cache, cmeta = load_fact_cache(fact_cache)
        hit = 0
        for i, rec in enumerate(recs):
            if rec["id"] in cache:
                facts[i] = merge_facts(cache[rec["id"]], facts[i])  # facts[i]=heuristic
                hit += 1
        judges: List[Dict[str, Dict[str, Any]]] = [{} for _ in recs]
        jhit = 0
        if judge_cache:
            jc = load_judge_cache(judge_cache)
            for i, rec in enumerate(recs):
                if rec["id"] in jc:
                    judges[i] = jc[rec["id"]]
                    jhit += 1
        rows, kept, dropped = build_rows(recs, pres, facts, indices, judges if jhit else None)
        write_csv(rows, out_path)
        errs = validate_csv(out_path, [r["id"] for r in recs])
        report = {
            "건수": len(recs), "모드": "사실캐시재생", "캐시적중": hit, "판정캐시적중": jhit,
            "캐시미스": len(recs) - hit, "캐시_LLM성공": cmeta.get("llm_done"),
            "전체_s": round(time.time() - t_all, 1),
            "근거_유지": kept, "근거_원문불일치": dropped,
            "양성수": {v: sum(int(r[v]) for r in rows) for v in ITEMS},
            "출력": out_path, "자가검증": "PASS" if not errs else errs,
        }
        log(json.dumps(report, ensure_ascii=False))
        return report

    # ── ② 모델 적재 ─────────────────────────────────────────────────────────
    llm_done, invalid = 0, 0
    inf_seconds = 0.0
    if runner_cls is None:                       # --no-llm : 규칙 엔진 단독 평가용
        errs = validate_csv(out_path, [r["id"] for r in recs])
        pos = {v: sum(int(r[v]) for r in rows) for v in ITEMS}
        log(json.dumps({"건수": len(recs), "LLM건수": 0, "양성수": pos, "출력": out_path,
                        "자가검증": "PASS" if not errs else errs}, ensure_ascii=False))
        return {"건수": len(recs), "LLM건수": 0, "출력": out_path,
                "자가검증": "PASS" if not errs else errs}
    # ── ②-a 항목→조문 사전 매핑(방안 A) : vLLM 적재 전에 bge-m3 로 1회 계산하고 GPU 메모리를 반납한다.
    #   조문은 1차 사실추출에는 넣지 않는다(Colab 측정: 효과 0). 2차 판정(⑥) 프롬프트에만 사용.
    item_map: Optional["ItemLawMap"] = None
    if not no_judge:
        try:
            rag = LawRAG.build(data_dir)
            item_map = ItemLawMap(rag)
            rag.release()
            del rag
        except Exception as e:
            log(f"항목→조문 매핑 생략(예외 {type(e).__name__}: {e}) → 2차 판정은 조문 없이 진행")
            item_map = None

    try:
        runner = runner_cls(FACT_SCHEMA, **runner_kw)
    except Exception as e:
        log(f"[치명] 모델 적재 실패 → 규칙 기반 답안 유지: {type(e).__name__}: {e}")
        errs = validate_csv(out_path, [r["id"] for r in recs])
        return {"건수": len(recs), "LLM건수": 0, "출력": out_path,
                "자가검증": "PASS" if not errs else errs}
    log(f"모델 로드 {runner.load_seconds:.1f}s (경과 {time.time() - t_start:.0f}s)")

    # ── ③ 프롬프트 구성 ─────────────────────────────────────────────────────
    reserve = getattr(runner, "max_tokens", MAX_TOKENS) + 256
    msgs_all, ntok = [], []
    for rec, pre in zip(recs, pres):
        try:
            m = truncate_messages(build_messages(rec, pre), runner, reserve)
        except Exception as e:
            log(f"  ! {rec['id']} 프롬프트 구성 실패: {type(e).__name__}: {e}")
            m = [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": clip(notice_text(rec), 3000)}]
        msgs_all.append(m)
        ntok.append(runner.count_tokens(m))
    log(f"프롬프트 토큰 중앙값 {sorted(ntok)[len(ntok) // 2]:,} · 최대 {max(ntok):,}")

    # ── ④ 청크 추론 + 데드라인 컨트롤러 ──────────────────────────────────────
    t_inf = time.time()
    done = 0
    chunk = max(1, chunk)
    while done < len(recs):
        elapsed = time.time() - t_start
        if done:
            per = (time.time() - t_inf) / done
            need = per * min(chunk, len(recs) - done) * 1.25 + 90
            if elapsed + need > deadline:
                log(f"[데드라인] 경과 {elapsed:.0f}s · 잔여 {len(recs) - done}건은 규칙 기반 답안 유지")
                break
        s, e = done, min(done + chunk, len(recs))
        texts = safe_chat(runner, msgs_all[s:e])
        for i, txt in enumerate(texts, start=s):
            f, ok = parse_facts(txt)
            if ok:
                facts[i] = merge_facts(f, facts[i])   # facts[i]=heuristic 기반, 신뢰필드만 덮음
                llm_done += 1
            else:
                invalid += 1                                  # 규칙 기반 사실 유지
        done = e
        log(f"  추론 {done}/{len(recs)} · {time.time() - t_inf:.0f}s · 총경과 {time.time() - t_start:.0f}s")
        if done % (chunk * 4) == 0 or done == len(recs):      # 중간 저장
            rows, _, _ = build_rows(recs, pres, facts, indices)
            write_csv(rows, out_path)
    inf_seconds = time.time() - t_inf

    if dump_facts:
        dump_fact_cache(dump_facts, recs, facts, llm_done, done)

    # 1차 결과(규칙+사실)를 먼저 저장 — 2차 판정 중 사고가 나도 이 답안이 남는다.
    rows, _, _ = build_rows(recs, pres, facts, indices)
    write_csv(rows, out_path)

    # ── ⑥ 2차 판정(캐스케이드) : 항목별 조문 + 판정기준 + 발췌 → {v, conf, q} ─────────
    #   규칙이 발화한 항목(억제 검증)과 약한 게이트만 걸린 항목(복구 후보)만 질의한다.
    #   우선순위(JUDGE_PRIORITY) 순으로 정렬해 데드라인 절단 시 가치 높은 질의가 먼저 처리된다.
    judges: List[Dict[str, Dict[str, Any]]] = [{} for _ in recs]
    jq_total, jq_done, jq_invalid, judge_seconds = 0, 0, 0, 0.0
    jq_split, jq_abstain = 0, 0        # [추가] 샘플 불일치·logprob 기권 건수(보고용)
    if not no_judge:
        t_j = time.time()
        queue: List[Tuple[int, int, str]] = []            # (우선순위, 레코드idx, 항목)
        prio = {v: k for k, v in enumerate(JUDGE_PRIORITY)}
        for i, (rec, pre, f) in enumerate(zip(recs, pres, facts)):
            try:
                res = apply_rules(f, rec, pre)
                for v in judge_candidates(res, f, rec, pre):
                    queue.append((prio.get(v, 99), i, v))
            except Exception as e:
                log(f"  ! {rec['id']} 판정 후보 산출 실패: {type(e).__name__}: {e}")
        queue.sort()
        jq_total = len(queue)
        log(f"2차 판정 질의 {jq_total}건 (레코드당 {jq_total / max(1, len(recs)):.2f}) · "
            f"항목별 {dict(Counter(v for _, _, v in queue))}")
        try:
            runner.set_judge(JUDGE_SCHEMA, JUDGE_MAX_TOKENS)
        except Exception as e:
            log(f"  ! 2차 판정 파라미터 설정 실패 → 2차 판정 생략: {type(e).__name__}: {e}")
            queue = []
        jreserve = JUDGE_MAX_TOKENS + 256
        jchunk = max(1, chunk * 2)
        pos_q = 0
        while pos_q < len(queue):
            elapsed = time.time() - t_start
            if jq_done:
                per = (time.time() - t_j) / jq_done
                need = per * min(jchunk, len(queue) - pos_q) * 1.25 + 60
                if elapsed + need > deadline:
                    log(f"[데드라인] 경과 {elapsed:.0f}s · 2차 판정 잔여 {len(queue) - pos_q}건 생략")
                    break
            batch_items = queue[pos_q:pos_q + jchunk]
            batch_msgs = []
            for _, i, v in batch_items:
                try:
                    law_ctx = item_map.get(v, pres[i]["local"]) if item_map is not None else ""
                    m = truncate_messages(build_judge_messages(v, recs[i], pres[i], law_ctx), runner, jreserve)
                except Exception as e:
                    log(f"  ! {recs[i]['id']}/{v} 판정 프롬프트 실패: {type(e).__name__}: {e}")
                    m = [{"role": "system", "content": JUDGE_SYSTEM},
                         {"role": "user", "content": f"## 점검항목 {v}\n" + ITEM_CRITERIA[v]}]
                batch_msgs.append(m)
            # [변경] F/G: 프롬프트당 JUDGE_SAMPLES 개 샘플 → 다수결 + logprob 기권으로 1건 판정 생성
            sample_sets = safe_chat_judge_n(runner, batch_msgs)
            for (_, i, v), samples in zip(batch_items, sample_sets):
                j = merge_judge_samples(samples)
                if j is None:
                    jq_invalid += 1
                else:
                    judges[i][v] = j
                    jq_done += 1
                    if j.get("agree", 1) < j.get("n", 1):
                        jq_split += 1
                    if j.get("margin") is not None and j["margin"] < JUDGE_MARGIN_MIN:
                        jq_abstain += 1
            pos_q += len(batch_items)
            log(f"  판정 {pos_q}/{len(queue)} · {time.time() - t_j:.0f}s · 총경과 {time.time() - t_start:.0f}s")
            if pos_q % (jchunk * 4) == 0 or pos_q == len(queue):
                rows, _, _ = build_rows(recs, pres, facts, indices, judges)
                write_csv(rows, out_path)
        judge_seconds = time.time() - t_j
        if dump_judge:
            dump_judge_cache(dump_judge, recs, judges, jq_done, jq_total)

    # ── ⑤ 최종 판정·근거·저장 ───────────────────────────────────────────────
    rows, kept, dropped = build_rows(recs, pres, facts, indices, judges)
    assert len(rows) == len(recs)
    write_csv(rows, out_path)
    errs = validate_csv(out_path, [r["id"] for r in recs])

    pos = {v: sum(int(r[v]) for r in rows) for v in ITEMS}
    n_sup = sum(1 for jd in judges for j in jd.values() if j.get("v") == 0 and j.get("conf") == "high")
    n_rec = sum(1 for jd in judges for j in jd.values() if j.get("v") == 1 and j.get("conf") == "high")
    report = {
        "건수": len(recs), "LLM성공": llm_done, "JSON실패": invalid,
        "판정질의": jq_total, "판정성공": jq_done, "판정JSON실패": jq_invalid,
        "판정샘플": JUDGE_SAMPLES, "판정불일치": jq_split, "판정기권": jq_abstain,
        "판정_high_0": n_sup, "판정_high_1": n_rec,
        "모델로드_s": round(getattr(runner, "load_seconds", 0.0), 1),
        "추론_s": round(inf_seconds, 1), "판정_s": round(judge_seconds, 1),
        "건당_s": round(inf_seconds / max(1, done), 2),
        "판정건당_s": round(judge_seconds / max(1, jq_done + jq_invalid), 3),
        "전체_s": round(time.time() - t_all, 1),
        "근거_유지": kept, "근거_원문불일치": dropped,
        "양성수": pos, "출력": out_path,
        "자가검증": "PASS" if not errs else errs,
    }
    log(json.dumps(report, ensure_ascii=False))
    return report


def main() -> int:
    t_start = time.time()
    ap = argparse.ArgumentParser(description="나라장터 자체입찰 공고 법령 위반사항 판정")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--output-dir", default=OUTPUT_DIR)
    ap.add_argument("--input", default=None, help="기본 = <data-dir>/test.jsonl.gz")
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--quantization", default=os.environ.get("PPS_QUANT", QUANT),
                    help="채점 서버 = int8_per_channel_weight_only · 'none'이면 미양자화")
    ap.add_argument("--gpu-mem", type=float, default=0.92)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=96, help="LLM.chat 한 번에 넘길 건수")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--mock", action="store_true", help="모델 없이 흐름·제출형식만 확인")
    ap.add_argument("--no-llm", action="store_true", help="규칙 엔진만으로 채점(개발용 하한 측정)")
    ap.add_argument("--dump-facts", default="",
                    help="[개발용] LLM 이 채운 사실을 JSONL 로 저장 (GPU 환경에서 1회 수집)")
    ap.add_argument("--facts", default="",
                    help="[개발용] 저장해 둔 사실 캐시를 주입하고 LLM 을 건너뜀 — "
                         "규칙 엔진 변경을 GPU 없이 실제 LLM 출력으로 평가")
    ap.add_argument("--dump-judge", default="",
                    help="[개발용] 2차 판정 결과를 JSONL 로 저장 (GPU 환경에서 1회 수집)")
    ap.add_argument("--judge", default="",
                    help="[개발용] 저장해 둔 2차 판정 캐시를 주입 (--facts 와 병용)")
    ap.add_argument("--no-judge", action="store_true", help="2차 판정(캐스케이드) 비활성화")
    ap.add_argument("--all-policy", action="store_true",
                    help="[개발용] 전 항목 suppress+recover 후보를 질의해 판정·logprob 캐시를 수집(정책 튜닝용). "
                         "이때의 submission.csv 는 무의미")
    ap.add_argument("--deadline", type=float, default=float(os.environ.get("PPS_DEADLINE", 6300)),
                    help="프로세스 시작 후 이 시간(초)이 지나면 남은 건은 규칙 기반으로 마감")
    a = ap.parse_args()
    a.data_dir = resolve_data_dir(a.data_dir)
    if a.all_policy:
        JUDGE_POLICY.clear()
        JUDGE_POLICY.update({v: {"suppress": "high", "recover": "high"} for v in ITEMS})
        globals()["JUDGE_MAX_PER_REC"] = 24
        log(f"[all-policy] 판정 후보 전수 질의 모드 · logprobs={JUDGE_LOGPROBS} · margin_min={JUDGE_MARGIN_MIN}")
    else:
        # model/ 정적 자산에 튜닝된 판정 정책이 있으면 채택(없으면 내장 기본값 · 회귀 0).
        used = load_policy_asset()
        if used:
            log(f"[asset] 판정 정책 자산 적용: {used} · 항목 {sorted(JUDGE_POLICY, key=lambda x: int(x[1:]))}")

    input_path = a.input
    if not input_path:
        for name in ("test.jsonl.gz", "test.jsonl"):
            p = os.path.join(a.data_dir, name)
            if os.path.exists(p):
                input_path = p
                break
        input_path = input_path or os.path.join(a.data_dir, "test.jsonl.gz")
    out_path = os.path.join(a.output_dir, "submission.csv")
    quant = None if str(a.quantization).lower() in ("none", "") else a.quantization
    runner_kw: Dict[str, Any] = {} if a.mock else dict(
        model_dir=a.model_dir, quant=quant, max_tokens=a.max_tokens, seed=SEED,
        gpu_mem=a.gpu_mem, tp=a.tp, max_model_len=a.max_model_len)

    try:
        cls = None if a.no_llm else (MockRunner if a.mock else VLLMRunner)
        if a.no_llm:
            runner_kw = {}
        report = run(input_path, out_path, cls,
                     data_dir=a.data_dir, limit=a.limit, chunk=a.chunk,
                     deadline=a.deadline, t_start=t_start,
                     dump_facts=a.dump_facts, fact_cache=a.facts,
                     dump_judge=a.dump_judge, judge_cache=a.judge, no_judge=a.no_judge, **runner_kw)
    except Exception as e:                       # 최후 방어: 빈 답안이라도 남긴다
        log(f"[치명] {type(e).__name__}: {e}")
        traceback.print_exc()
        try:
            ids = [r["id"] for r in iter_records(input_path, limit=a.limit)]
            write_csv([empty_row(i) for i in ids], out_path)
            log(f"빈 답안 {len(ids)}건 기록 → {out_path}")
        except Exception as e2:
            log(f"[치명] 빈 답안 기록도 실패: {type(e2).__name__}: {e2}")
        return 1

    log(f"총 소요 {time.time() - t_start:.0f}s")
    return 0 if report.get("자가검증") in ("PASS", None) else 1


if __name__ == "__main__":
    sys.exit(main())
