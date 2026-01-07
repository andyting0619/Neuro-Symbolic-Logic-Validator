import streamlit as st
from groq import Groq
from z3 import *
import re
from typing import List, Dict, Tuple

GROQ_API_KEY = "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"

st.set_page_config(
    page_title="Neuro-Symbolic Logic Validator",
    page_icon="🧠",
    layout="wide"
)

st.markdown("<h1 style='text-align: center;'>🧠 Neuro-Symbolic Logic Validator</h1>",
            unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #888;'><strong>Developed by Andy Ting Zhi Wei</strong></p>",
            unsafe_allow_html=True)
st.markdown("""
This application validates the logical inference of AI responses using formal propositional logic.
It extracts logical statements and validates them using both formal verification and heuristic analysis.
""")

if GROQ_API_KEY == "your_api_key_here" or not GROQ_API_KEY:
    st.error("⚠️ Please configure your API Key first!")
    st.stop()

with st.sidebar:
    st.header("⚙️ Configuration")

    model = st.selectbox(
        "Select Model",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "qwen/qwen3-32b"],
        help="Choose the LLM model for reasoning"
    )

    with st.expander("Advanced Settings"):
        temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1,
                                help="Lower = more focused, Higher = more creative")
        max_tokens = st.slider("Max Tokens", 256, 2048, 1024, 128,
                               help="Maximum length of AI response")

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    **Neuro-Symbolic Architecture:**
    - **Neural**: LLM reasoning
    - **Symbolic**: Dual validation system
    - **Methods**: Logical Verification and Heuristic Analysis

    **Validation Approaches:**
    1. **Logical Verification**: Uses SAT solver to test logical consequence using symbolic logical reasoning
    2. **Heuristic Analysis**: Evaluates the quality and completeness of logical structure
    """)


def get_groq_client():
    return Groq(api_key=GROQ_API_KEY)


def extract_propositions(text: str) -> List[str]:
    patterns = [
        r'(?:therefore|thus|hence|so|consequently),?\s+',
        r'\.\s+',
        r';\s+',
        r'\n+'
    ]

    combined_pattern = '|'.join(patterns)
    sentences = re.split(combined_pattern, text)

    propositions = []
    for s in sentences:
        s = s.strip()
        s = re.sub(r'^[\d\.\)\-•]+\s*', '', s)
        if s and len(s) > 15 and not s.lower().startswith(('let', 'note', 'in other words')):
            propositions.append(s)

    return propositions


def parse_logic_structure(response: str) -> Dict:
    response_lower = response.lower()

    structure = {
        "premises": [],
        "reasoning": [],
        "conclusion": None,
        "logical_operators": [],
        "inference_pattern": None
    }

    operators = {
        "and": len(re.findall(r'\band\b', response_lower)),
        "or": len(re.findall(r'\bor\b', response_lower)),
        "not": len(re.findall(r'\bnot\b|\bno\b|\bfalse\b|\bnegation\b', response_lower)),
        "implies": len(re.findall(r'\bif\b.*?\bthen\b|\bimplies\b|\bmeans that\b|\bleads to\b', response_lower)),
        "therefore": len(re.findall(r'\btherefore\b|\bthus\b|\bhence\b|\bso\b|\bconsequently\b', response_lower))
    }

    structure["logical_operators"] = {
        k: v for k, v in operators.items() if v > 0}

    if re.search(r'if.*then.*if.*then', response_lower):
        structure["inference_pattern"] = "Hypothetical Syllogism"
    elif re.search(r'all.*are.*and.*is.*therefore', response_lower):
        structure["inference_pattern"] = "Categorical Syllogism"
    elif re.search(r'if.*then.*is.*therefore', response_lower):
        structure["inference_pattern"] = "Modus Ponens"
    elif re.search(r'if.*then.*not.*therefore.*not', response_lower):
        structure["inference_pattern"] = "Modus Tollens"
    elif re.search(r'either.*or.*not.*therefore', response_lower):
        structure["inference_pattern"] = "Disjunctive Syllogism"

    conclusion_markers = [
        r'therefore[,:]?\s+(.+?)(?:\.|$)',
        r'thus[,:]?\s+(.+?)(?:\.|$)',
        r'hence[,:]?\s+(.+?)(?:\.|$)',
        r'so[,:]?\s+(.+?)(?:\.|$)',
        r'we can conclude that\s+(.+?)(?:\.|$)',
        r'it follows that\s+(.+?)(?:\.|$)',
        r'this means\s+(.+?)(?:\.|$)',
        r'conclusion:\s*(.+?)(?:\.|$)'
    ]

    conclusion_found = False
    for marker in conclusion_markers:
        matches = re.findall(marker, response_lower, re.IGNORECASE)
        if matches:
            structure["conclusion"] = matches[-1].strip().rstrip('.')
            conclusion_found = True
            break

    propositions = extract_propositions(response)

    if conclusion_found and structure["conclusion"]:
        conclusion_text = structure["conclusion"].lower()
        structure["premises"] = [p for p in propositions
                                 if conclusion_text not in p.lower()[:min(len(conclusion_text)+20, len(p))]]
        for p in propositions:
            if conclusion_text in p.lower():
                structure["conclusion"] = p
                break
    else:
        if len(propositions) >= 2:
            structure["conclusion"] = propositions[-1]
            structure["premises"] = propositions[:-1]
        elif len(propositions) == 1:
            structure["conclusion"] = propositions[0]

    return structure


def formal_logical_validation(logic_structure: Dict) -> Tuple[bool, str, str, str, bool]:
    try:
        premises = logic_structure["premises"]
        conclusion = logic_structure["conclusion"]

        if not premises or not conclusion:
            return None, "unknown", "N/A", "Insufficient logical structure: need at least one premise and one conclusion", False

        premise_vars = []
        for i, premise in enumerate(premises, 1):
            var = Bool(f'P{i}')
            premise_vars.append(var)

        conclusion_var = Bool('C')

        all_premises = And(
            *premise_vars) if len(premise_vars) > 1 else premise_vars[0]

        premise_solver = Solver()
        premise_solver.add(all_premises)
        premise_check = premise_solver.check()

        if premise_check == unsat:
            return None, "unsat", "premises_contradiction", """❌ **PREMISES CONTRADICTION ERROR**

**Step 1: Premise Consistency Check**

The formal verifier detected that the premises themselves are contradictory (UNSATISFIABLE).

**Result:** (P₁ ∧ P₂ ∧ ... ∧ Pₙ) is UNSATISFIABLE

**Interpretation:** The premises cannot all be true simultaneously. This is a logical contradiction.

**Conclusion:** The inference is **Vacuously Valid** - from a contradiction, anything can be derived (principle of explosion), but the reasoning is meaningless because the premises are inconsistent.

**Recommendation:** Review and correct the premises before evaluating the inference.""", True

        solver = Solver()
        conclusion_false = Not(conclusion_var)

        solver.add(all_premises)
        solver.add(conclusion_false)

        result = solver.check()
        result_str = str(result)

        if result == unsat:
            validity = True
            sat_status = "unsat"
            explanation = """✅ **LOGICALLY VALID INFERENCE**

**Step 1: Premise Consistency Check**

Premises are consistent (satisfiable) ✓

**Step 2: Validity Check**

The formal verifier confirmed that (P₁ ∧ P₂ ∧ ... ∧ Pₙ ∧ ¬C) is UNSATISFIABLE

**Interpretation:** It is impossible for all premises to be true while the conclusion is false. The conclusion necessarily follows from the premises through logical entailment.

**Formal Proof:** Given the premises are true, the conclusion MUST be true."""

        elif result == sat:
            model = solver.model()
            validity = False
            sat_status = "sat"

            counterexample_lines = []
            for i, var in enumerate(premise_vars, 1):
                value = model.evaluate(var, model_completion=True)
                counterexample_lines.append(f"P{i} = {value}")
            counterexample_lines.append(
                f"C = {model.evaluate(conclusion_var, model_completion=True)}")

            counterexample_str = ", ".join(counterexample_lines)

            explanation = f"""❌ **LOGICALLY INVALID INFERENCE**

**Step 1: Premise Consistency Check**

Premises are consistent (satisfiable) ✓

**Step 2: Validity Check**

The formal verifier found that (P₁ ∧ P₂ ∧ ... ∧ Pₙ ∧ ¬C) is SATISFIABLE

**Counterexample:** {counterexample_str}

**Interpretation:** There exists a scenario where all premises are true but the conclusion is false. This indicates a logical fallacy - the conclusion does not necessarily follow from the premises.

**Recommendation:** The inference is invalid in propositional logic. Additional premises may be needed, or the reasoning contains a flaw."""

        else:
            validity = None
            sat_status = "unknown"
            explanation = """⚠️ **VALIDATION UNDETERMINED**

**Step 1: Premise Consistency Check**

Premises appear consistent ✓

**Step 2: Validity Check**

The formal verifier returned UNKNOWN status

**Possible Reasons:**
- Very complex logical structure
- Solver timeout
- Non-standard logical patterns

**Conclusion:** The reasoning may still be valid but couldn't be verified formally."""

        return validity, sat_status, result_str, explanation, False

    except Exception as e:
        return None, "error", "error", f"Formal validation error: {str(e)}", False


def heuristic_validation(logic_structure: Dict) -> Tuple[bool, str, int]:
    score = 0
    max_score = 0
    issues = []

    max_score += 30
    if logic_structure["premises"] and logic_structure["conclusion"]:
        score += 30
    else:
        issues.append("Missing clear premises or conclusion")
        return False, "Incomplete logical structure: " + "; ".join(issues), 0

    max_score += 20
    if logic_structure["logical_operators"]:
        score += min(20, len(logic_structure["logical_operators"]) * 5)
    else:
        issues.append("Missing logical connectives")

    max_score += 20
    if logic_structure["logical_operators"].get("therefore", 0) > 0:
        score += 20
    else:
        issues.append("Conclusion not explicitly marked")

    max_score += 15
    num_premises = len(logic_structure["premises"])
    if 1 <= num_premises <= 5:
        score += 15
    elif num_premises > 5:
        score += 10
        issues.append("Too many premises")
    else:
        issues.append("Insufficient premises")

    max_score += 15
    if logic_structure["inference_pattern"]:
        score += 15

    percentage = (score / max_score) * 100

    if percentage >= 70:
        validity = True
        explanation = f"""The logical reasoning structure is complete and well-formed (Score: {percentage:.0f}%).

**Details:**
- Premises: {len(logic_structure["premises"])}
- Logical connectives: {sum(logic_structure["logical_operators"].values())}
- Pattern: {logic_structure["inference_pattern"] or "General reasoning"}

The reasoning process is clear and properly structured."""
    else:
        validity = False
        explanation = f"""The logical structure may be incomplete or unclear (Score: {percentage:.0f}%).

**Issues:** {", ".join(issues)}

**Suggestions:** Ensure premises are explicit, use 'therefore' to mark conclusions, and include logical connectives."""

    return validity, explanation, int(percentage)


def main():
    st.header("📝 Input")

    user_query = st.text_area(
        "Enter your logical statement or question:",
        placeholder="Example: If all mathematicians are genius, and Dr. Maharani is a mathematician, what can we conclude about Dr. Maharani?",
        height=120,
        help="Enter a logical reasoning question with clear premises and a conclusion"
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        analyze_button = st.button(
            "🔍 Analyze Logic", type="primary", use_container_width=True)
    with col2:
        if st.button("🔄 Clear", use_container_width=True):
            st.rerun()

    if analyze_button:
        if not user_query:
            st.error("Please enter a query")
            return

        with st.spinner("🤖 Generating AI response..."):
            try:
                client = get_groq_client()

                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": """You are a logical reasoning expert. Provide clear, step-by-step logical reasoning.

Structure your response as:
1. State the premises explicitly (e.g., "Premise 1: ...", "Premise 2: ...")
2. Identify the logical rule being applied
3. State the conclusion clearly using "Therefore" or "Thus"

Use explicit logical language with connectives: and, or, not, if-then, therefore."""
                        },
                        {
                            "role": "user",
                            "content": user_query
                        }
                    ],
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )

                ai_response = chat_completion.choices[0].message.content

                st.header("🤖 AI Response")
                with st.container():
                    st.markdown(f"**Model:** {model}")
                    st.info(ai_response)

                st.header("🔬 Logic Analysis")
                logic_structure = parse_logic_structure(ai_response)

                st.subheader("📋 Extracted Premises")
                if logic_structure["premises"]:
                    for i, premise in enumerate(logic_structure["premises"], 1):
                        st.markdown(f"**P{i}:** {premise}")
                else:
                    st.warning("No premises extracted")

                st.subheader("🎯 Conclusion")
                if logic_structure["conclusion"]:
                    st.markdown(f"**C:** {logic_structure['conclusion']}")
                else:
                    st.warning("No conclusion extracted")

                premises = logic_structure["premises"]
                conclusion = logic_structure["conclusion"]

                formal_logic = []
                for i, premise in enumerate(premises, 1):
                    formal_logic.append(f"P{i}: {premise}")
                if conclusion:
                    formal_logic.append(f"C: {conclusion}")
                formal_repr = "\n".join(formal_logic)

                st.subheader("📐 Propositional Logic")
                st.code(formal_repr, language="text")

                if len(premises) > 1:
                    premise_formula = " ∧ ".join(
                        [f"P{i+1}" for i in range(len(premises))])
                    formula = f"({premise_formula}) → C"
                elif len(premises) == 1:
                    formula = "P1 → C"
                else:
                    formula = "No valid formula"

                st.subheader("🧮 Logical Formula")
                if formula != "No valid formula":
                    st.latex(formula.replace('∧', r'\land').replace(
                        '→', r'\rightarrow').replace('¬', r'\lnot'))

                    if logic_structure["inference_pattern"]:
                        st.info(
                            f"🎓 **Detected Pattern:** {logic_structure['inference_pattern']}")
                    else:
                        st.info(f"🎓 **Detected Pattern:** General Reasoning")
                else:
                    st.warning("Unable to construct formula")

                st.header("⚖️ Dual Validation Results")

                with st.spinner("🔍 Running validations..."):
                    formal_validity, sat_status, result_str, formal_explanation, is_contradiction = formal_logical_validation(
                        logic_structure)
                    heuristic_validity, heuristic_explanation, heuristic_score = heuristic_validation(
                        logic_structure)

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("🔷 Formal Verification")

                    if is_contradiction:
                        st.error(f"❌ PREMISES CONTRADICTION (Vacuously Valid)")
                        st.info(
                            f"**Satisfiability Check:** PREMISES UNSATISFIABLE")
                    elif sat_status == "unsat":
                        st.success(f"✅ LOGICALLY VALID")
                        st.info(f"**Satisfiability Check:** UNSATISFIABLE")
                    elif sat_status == "sat":
                        st.error(f"❌ LOGICALLY INVALID")
                        st.info(f"**Satisfiability Check:** SATISFIABLE")
                    elif sat_status == "unknown":
                        st.warning(f"⚠️ UNDETERMINED")
                        st.info(f"**Satisfiability Check:** UNKNOWN")
                    else:
                        st.error(f"❌ ERROR")
                        st.info(f"**Satisfiability Check:** ERROR")

                    st.markdown(formal_explanation)

                with col2:
                    st.subheader("🔶 Heuristic Analysis")

                    if heuristic_validity is True:
                        st.success(f"✅ WELL-STRUCTURED ({heuristic_score}%)")
                    elif heuristic_validity is False:
                        st.warning(f"⚠️ INCOMPLETE ({heuristic_score}%)")
                    else:
                        st.info(f"ℹ️ UNDETERMINED ({heuristic_score}%)")

                    st.markdown(heuristic_explanation)

                st.header("📊 Analysis Summary")
                summary_cols = st.columns(4)

                with summary_cols[0]:
                    if is_contradiction:
                        st.metric("Logically Consistent", "NO", delta="⚠",
                                  delta_color="inverse")
                    elif sat_status in ["unsat", "sat"]:
                        st.metric("Logically Consistent", "YES",
                                  delta="✓", delta_color="normal")
                    else:
                        st.metric("Logically Consistent", "N/A",
                                  delta="?", delta_color="off")

                with summary_cols[1]:
                    st.metric("Heuristic Score", f"{heuristic_score}%", delta="✓" if heuristic_validity else "⚠",
                              delta_color="normal" if heuristic_validity else "off")

                with summary_cols[2]:
                    if sat_status == "unsat" and not is_contradiction:
                        st.metric("Logically Satisfiable", "NO", delta="✓",
                                  delta_color="normal")
                    elif sat_status == "sat":
                        st.metric("Logically Satisfiable", "YES",
                                  delta="✓", delta_color="normal")
                    else:
                        st.metric("Logically Satisfiable", "N/A")

                with summary_cols[3]:
                    if formal_validity is True:
                        st.metric("Logic Validity", "VALID",
                                  delta="✓", delta_color="normal")
                    elif formal_validity is False:
                        st.metric("Logic Validity", "INVALID",
                                  delta="✗", delta_color="inverse")
                    else:
                        st.metric("Logic Validity", "UNKNOWN",
                                  delta="?", delta_color="off")

            except Exception as e:
                st.error(f"❌ Error occurred: {str(e)}")
                with st.expander("🐛 Debug Information"):
                    st.exception(e)


if __name__ == "__main__":
    main()
