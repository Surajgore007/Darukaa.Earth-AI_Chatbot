"""
Grounding Validator Node.

Strictly validates that every numeric claim and source citation in the generated response
is directly and completely supported by the retrieved evidence.
Rejects hallucinations and retries once before falling back to:
'Insufficient evidence for a confident recommendation.'
"""

import re
from typing import Dict, Any, List
from app.graph.state import GraphState


def normalize_numeric_string(s: str) -> str:
    """
    Standardizes dashes, hyphens, and whitespace in numeric expressions.
    Example: '+15 – 25 %' -> '+15-25%'
    """
    s = s.replace("–", "-").replace("—", "-").replace("−", "-").replace("\u2011", "-")
    # Normalize unicode spaces (\u202f, \u00a0, etc.) to standard space
    s = re.sub(r"[\s\u202f\u00a0\u2000-\u200f\ufeff]+", " ", s)
    # Remove whitespace between digits, hyphens, and percent signs
    s = re.sub(r"\s*-\s*", "-", s)
    s = re.sub(r"\s*%", "%", s)
    return s.strip().lower()


def extract_numeric_expressions(text: str) -> List[str]:
    """
    Extracts complete numeric expressions, ranges, and units from text.
    Matches:
    - Ranges with percentages: '+15–25%', '20-40%', '-50 to 85%'
    - Single percentages: '0.3%', '25%'
    - Quantities with units: '2-3 years', '10-30 meters', '3.0°C', '0.2 to 3.1 t C/ha'
    """
    norm_text = text.replace("–", "-").replace("—", "-").replace("−", "-").replace("\u2011", "-")
    norm_text = re.sub(r"[\s\u202f\u00a0\u2000-\u200f\ufeff]+", " ", norm_text)
    
    patterns = [
        # Range with percentage: e.g. +15-25%, 20-35%, -50-80%
        r"[+-]?\d+(?:\.\d+)?\s*(?:-|to)\s*\d+(?:\.\d+)?\s*%",
        # Single percentage: e.g. 0.3%, 25%, +15%
        r"[+-]?\d+(?:\.\d+)?\s*%",
        # Range with unit: e.g. 2-3 years, 10-30 meters, 0.2 to 3.1 t c/ha
        r"\d+(?:\.\d+)?\s*(?:-|to)\s*\d+(?:\.\d+)?\s*(?:years?|meters?|°c|t\s*c/ha)",
        # Single unit: e.g. 3°C, 2 years
        r"\d+(?:\.\d+)?\s*(?:years?|meters?|°c|t\s*c/ha)"
    ]
    
    combined_regex = re.compile("|".join(patterns), re.IGNORECASE)
    matches = combined_regex.findall(norm_text)
    
    # Return unique normalized expressions
    unique_expressions = []
    for m in matches:
        norm_m = normalize_numeric_string(m)
        if norm_m not in unique_expressions:
            unique_expressions.append(norm_m)
    return unique_expressions


def validate_claim_attributions(draft: str) -> List[str]:
    """
    Validates source-to-claim and intervention-to-metric pairings:
    1. Metric != Downstream Outcome:
       - +20-35% microbial biomass must NOT be claimed as overall biodiversity gain or species richness.
       - +20-40% moisture retention must NOT be claimed as biodiversity improvement.
       - +15-25% SOC must NOT be claimed as pollinator diversity improvement.
    2. Attribution Integrity:
       - +25-50% pollinator richness must NOT be attributed to agroforestry (belongs to perennial hedgerows/floral margins).
       - -50-85% nitrate reduction must NOT be attributed to in-field agroforestry (belongs to riparian buffer strips).
    3. Exact Evidence Mapping (if EVIDENCE: section exists):
       - Each bullet must match its exact verified intervention -> metric -> source.
    """
    errors = []
    draft_lower = draft.lower()

    # Normalize unicode dashes for attribution regexes
    norm_draft = draft_lower.replace("–", "-").replace("—", "-").replace("−", "-").replace("\u2011", "-")

    # 1. Reject conflation of microbial biomass +20-35% with overall biodiversity or species richness
    if "20-35%" in norm_draft or "20 to 35%" in norm_draft:
        # Check lines where 20-35% appears
        for line in norm_draft.split("\n"):
            line_s = line.strip()
            if not ("20-35%" in line_s or "20 to 35%" in line_s):
                continue

            # Strip the known FAO report title so 'Soil Biodiversity' in the title doesn't trigger false positives
            clean_line = line_s.replace("state of knowledge of soil biodiversity", "fao_report")
            clean_line = clean_line.replace("soil biodiversity (2020)", "fao_report")
            clean_line = clean_line.replace("soil biodiversity", "fao_report")

            # If the line explicitly identifies microbial biomass, check if it also claims overall biodiversity gain
            if "microbial biomass" in clean_line or "microbial community" in clean_line:
                if re.search(r"(?:overall\s+)?biodiversity\s*(?:gain|increase|by|status|richness)?\s*[:=]\s*(?:\+)?20\s*[-–to\s]+\s*35%", clean_line):
                    errors.append("Microbial biomass +20-35% cannot be claimed as overall biodiversity gain")
            elif "biodiversity" in clean_line or "species richness" in clean_line:
                # Line has 20-35% and biodiversity/species richness WITHOUT microbial biomass!
                errors.append("Microbial biomass +20-35% cannot be claimed as overall biodiversity gain or species richness")

        # Check in IMPACTED METRICS: if bullet header is Biodiversity with +20-35%
        if "impacted metrics:" in norm_draft:
            metrics_section = norm_draft.split("impacted metrics:")[1].split("time horizon:")[0]
            for line in metrics_section.split("\n"):
                line_s = line.strip()
                clean_m_line = line_s.replace("soil biodiversity", "fao_report")
                if ("20-35%" in clean_m_line or "20 to 35%" in clean_m_line) and "biodiversity" in clean_m_line and "microbial" not in clean_m_line:
                    errors.append("Biodiversity listed with +20-35% under IMPACTED METRICS (belongs strictly to microbial biomass)")
                    break

    # 2. Reject conflation of soil moisture +20-40% with biodiversity
    if "20-40%" in norm_draft or "20 to 40%" in norm_draft:
        for line in norm_draft.split("\n"):
            clean_line = line.replace("soil biodiversity", "").strip()
            if ("20-40%" in clean_line or "20 to 40%" in clean_line) and "biodiversity" in clean_line and "moisture" not in clean_line:
                errors.append("Soil moisture +20-40% cannot be claimed as biodiversity improvement")

    # 3. Reject conflation of SOC +15-25% with pollinator diversity
    if "15-25%" in norm_draft or "15 to 25%" in norm_draft:
        for line in norm_draft.split("\n"):
            if ("15-25%" in line or "15 to 25%" in line) and ("pollinator" in line or "bee" in line) and "carbon" not in line:
                errors.append("Soil organic carbon +15-25% cannot be claimed as pollinator diversity improvement")

    # 4. Reject attribution of pollinator +25-50% to agroforestry
    if "25-50%" in norm_draft or "25 to 50%" in norm_draft:
        # Check IMPACTED METRICS: cannot list 25-50% without hedgerow/margin attribution
        if "impacted metrics:" in norm_draft:
            metrics_sec = norm_draft.split("impacted metrics:")[1].split("time horizon:")[0]
            for line in metrics_sec.split("\n"):
                if ("25-50%" in line or "25 to 50%" in line) and "hedgerow" not in line and "margin" not in line:
                    errors.append("Pollinator +25-50% is listed under impacted metrics without hedgerow attribution")
        # In text, reject direct attribution claiming agroforestry produces 25-50% pollinators
        for line in norm_draft.split("\n"):
            line_s = line.strip()
            if ("25-50%" in line_s or "25 to 50%" in line_s) and "agroforestry" in line_s:
                if "hedgerow" in line_s or "margin" in line_s or "does not provide" in line_s or "not agroforestry" in line_s:
                    continue
                if re.search(r"agroforestry[^\.\n]*?(?:increases?|provides?|results?|yields?)[^\.\n]*?(?:\+)?25\s*[-–to\s]+\s*50%", line_s):
                    errors.append("Pollinator +25-50% is misattributed directly to agroforestry")

    # 5. Reject attribution of nitrate runoff -50-85% to agroforestry
    if "50-85%" in norm_draft or "50 to 85%" in norm_draft:
        if "impacted metrics:" in norm_draft:
            metrics_sec = norm_draft.split("impacted metrics:")[1].split("time horizon:")[0]
            for line in metrics_sec.split("\n"):
                if ("50-85%" in line or "50 to 85%" in line) and "riparian" not in line and "buffer" not in line:
                    errors.append("Nitrate reduction -50-85% listed under impacted metrics without riparian buffer attribution")
        for line in norm_draft.split("\n"):
            line_s = line.strip()
            if ("50-85%" in line_s or "50 to 85%" in line_s) and "agroforestry" in line_s:
                if "riparian" in line_s or "buffer" in line_s or "does not establish" in line_s:
                    continue
                if re.search(r"agroforestry[^\.\n]*?(?:reduces?|intercepts?|filters?)[^\.\n]*?(?:-)?50\s*[-–to\s]+\s*85%", line_s):
                    errors.append("Nitrate reduction -50-85% is misattributed directly to agroforestry")

    # 6. Validate EVIDENCE block if present
    if "evidence:" in norm_draft:
        evidence_sec = norm_draft.split("evidence:")[1].split("impacted metrics:")[0]
        for line in evidence_sec.split("\n"):
            line = line.strip()
            if not line or not line.startswith("*"):
                continue
            # If line mentions 20-35%, it must be microbial biomass, not biodiversity
            if ("20-35%" in line or "20 to 35%" in line) and "biodiversity" in line and "microbial" not in line:
                errors.append("EVIDENCE block misattributes +20-35% to biodiversity instead of microbial biomass")
            # If line mentions 25-50%, it must cite IPBES (2019)
            if ("25-50%" in line or "25 to 50%" in line) and "ipbes" not in line:
                errors.append("EVIDENCE block for +25-50% must cite IPBES (2019)")
            # If line mentions 50-85%, it must cite Agriculture, Ecosystems & Environment
            if ("50-85%" in line or "50 to 85%" in line) and "agee" not in line and "agriculture, ecosystems" not in line:
                errors.append("EVIDENCE block for -50-85% must cite Agriculture, Ecosystems & Environment (2021)")

    return errors


def extract_cited_urls(text: str) -> List[str]:
    """
    Extracts URLs cited in the response text.
    """
    pattern = r"https?://[^\s)\]]+"
    return [url.rstrip("/.,)]") for url in re.findall(pattern, text)]


def validate_grounding(state: GraphState) -> Dict[str, Any]:
    """
    Validates the draft response against the retrieved evidence.
    Enforces strict complete numeric matching with zero loose-digit fallback,
    and validates intervention -> metric attribution integrity.
    """
    draft = state.get("draft_response", "")
    retrieved_data = state.get("retrieved_context", {})
    evidence_text = retrieved_data.get("combined_context_prompt", "")
    retry_count = state.get("retry_count", 0)

    # If the response is already an explicit fallback or clarification, pass through
    if not draft or "insufficient evidence" in draft.lower():
        return {
            "validation_passed": True,
            "final_response": draft
        }

    # Normalize evidence text for robust exact matching
    norm_evidence = normalize_numeric_string(evidence_text)

    # 1. Collect all user-supplied numbers across conversation history and variables
    user_strings = [state.get("current_user_message", "")]
    for val in state.get("variables", {}).values():
        if val:
            user_strings.append(str(val))
    for msg in state.get("messages", []):
        if msg.get("role") == "user":
            user_strings.append(msg.get("content", ""))

    all_user_text = " ".join(user_strings)
    user_numbers = [normalize_numeric_string(u) for u in extract_numeric_expressions(all_user_text)]

    # 2. Validate Numeric Claims: require complete expression/range match
    draft_numbers = extract_numeric_expressions(draft)
    unsupported_numbers = []

    for num in draft_numbers:
        # If the number was supplied by the user (e.g. 0.3% SOC from Turn 1/2), it is valid context
        if num in user_numbers:
            continue

        # Otherwise, the complete numeric range/expression MUST exist in retrieved evidence
        num_variants = [
            num,
            num.lstrip("+-"),
            f"+{num.lstrip('+-')}",
            f"-{num.lstrip('+-')}"
        ]
        
        is_supported = any(var in norm_evidence for var in num_variants)
        if not is_supported:
            unsupported_numbers.append(num)

    # 3. Validate Sources: every cited URL must exist in the retrieved evidence
    draft_urls = extract_cited_urls(draft)
    unsupported_urls = []
    for url in draft_urls:
        if url.lower() not in evidence_text.lower():
            unsupported_urls.append(url)

    # 4. Validate claim attributions and intervention-metric pairings
    attribution_errors = validate_claim_attributions(draft)

    # Grounding decision
    is_grounded = (len(unsupported_numbers) == 0 and len(unsupported_urls) == 0 and len(attribution_errors) == 0)

    if is_grounded:
        return {
            "validation_passed": True,
            "final_response": draft
        }

    # If validation failed:
    if retry_count < 1:
        # Retry once with stricter prompt
        print(f"[GROUNDING FAIL] Unsupported numbers: {unsupported_numbers}, Unsupported URLs: {unsupported_urls}, Attribution errors: {attribution_errors}. Retrying once...")
        return {
            "validation_passed": False,
            "retry_count": retry_count + 1
        }
    else:
        # Failed twice -> Return challenge-mandated failure response
        print(f"[GROUNDING FAIL] Second failure on grounding. Returning 'Insufficient evidence for a confident recommendation.'")
        return {
            "validation_passed": True,
            "final_response": "Insufficient evidence for a confident recommendation."
        }
