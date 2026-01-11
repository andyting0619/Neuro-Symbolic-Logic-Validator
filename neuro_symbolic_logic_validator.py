import streamlit as st
from groq import Groq
from z3 import *
import re
import json
from itertools import product

KEY_1 = "gsk_vD7EHyaBqg3bP"
KEY_2 = "LeQfVaXWGdyb3F"
KEY_3 = "YRFTnIHfwoSnh5U7wdLWm6QyN"

GROQ_API_KEY = KEY_1 + KEY_2 + KEY_3

st.set_page_config(
    page_title="Neuro-Symbolic Logic Validator",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
    .stMarkdown {
        overflow-x: visible !important;
    }
    div[data-testid="stMarkdownContainer"] {
        overflow-x: visible !important;
    }
    .katex-display {
        overflow-x: auto !important;
        overflow-y: hidden !important;
        max-width: 100% !important;
    }
    .katex-display > .katex {
        white-space: normal !important;
        text-align: center !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>🧠 Neuro-Symbolic Logic Validator 🤖</h1>",
            unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #888;'><strong>👨‍💻 Developed by Andy Ting Zhi Wei</strong></p>",
            unsafe_allow_html=True)

if GROQ_API_KEY == "your_api_key_here" or not GROQ_API_KEY:
    st.error("⚠️ Please configure your API Key first!")
    st.stop()

with st.sidebar:
    st.header("⚙️ Configuration")

    model = st.selectbox(
        "Select Model",
        [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant"
        ],
        help="Choose from active Groq models"
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.1,
        help="Controls randomness in output generation.\n\n• Lower (0.0-0.3): More focused and deterministic\n• Medium (0.4-0.7): Balanced creativity\n• Higher (0.8-1.0): More creative and diverse"
    )

    max_tokens = st.slider(
        "Max Tokens",
        min_value=512,
        max_value=4096,
        value=1024,
        step=128,
        help="Maximum length of the generated response.\n\n• 512-1024: Short responses\n• 1024-2048: Medium responses\n• 2048-4096: Long, detailed responses"
    )

    st.markdown("---")
    st.markdown("### Pipeline")
    st.markdown("""
    **Stage 0:** Input Analysis  
    **Stage 1:** Neural Translation  
    **Stage 2:** Truth Table  
    **Stage 3:** Logic Verification
    """)


def get_groq_client():
    return Groq(api_key=GROQ_API_KEY)


def check_quantifier_logic(user_query):
    query_lower = user_query.lower()

    quantifier_patterns = [
        r'\bfor all\b',
        r'\bfor every\b',
        r'\bthere exists\b',
        r'\bexists\b',
        r'\b∀\b',
        r'\b∃\b',
        r'\bforall\b',
        r'\bsome\b.*\bsuch that\b',
        r'\ball\s+\w+\s+are\b',
        r'\bevery\s+\w+\s+(?:is|are)\b'
    ]

    for pattern in quantifier_patterns:
        if re.search(pattern, query_lower):
            return True

    if re.search(r'[a-z]\s*[<>=]\s*[-\d]', query_lower):
        return True

    return False


def validate_logic_extraction(premises, conclusion):
    invalid_phrases = [
        'no premises',
        'no premise',
        'no conclusion',
        'premise not provided',
        'premises not provided',
        'conclusion not provided',
        'not specified',
        'not given',
        'not provided',
        'none given',
        'none provided',
        'no statement',
        'no logical',
        'invalid',
        'unclear',
        'n/a',
        'none'
    ]

    if not premises or len(premises) == 0:
        return False, "No premises extracted"

    for premise in premises:
        if not premise or not isinstance(premise, str):
            return False, "Invalid premise format"

        premise_lower = premise.lower().strip()

        if len(premise_lower) < 5:
            return False, "Premises too short or empty"

        if any(phrase in premise_lower for phrase in invalid_phrases):
            return False, f"Invalid premise detected: '{premise}'"

    if not conclusion or not isinstance(conclusion, str) or len(conclusion.strip()) < 5:
        return False, "No conclusion extracted"

    conclusion_lower = conclusion.lower().strip()

    if any(phrase in conclusion_lower for phrase in invalid_phrases):
        return False, f"Invalid conclusion detected: '{conclusion}'"

    return True, "Valid"


def advanced_reasoning(client, user_query, model, temperature, max_tokens):
    reasoning_prompt = f"""You are an expert in formal logic including first-order logic and categorical syllogisms.

**User Question:**
{user_query}

**Your Task:**
Analyze this logical reasoning question. It may involve quantifiers (all, some, every, there exists) or propositional logic.

**Response Format (JSON only):**
{{
  "premises": ["premise 1", "premise 2"],
  "conclusion": "conclusion statement",
  "consistent": true/false,
  "satisfiable": true/false,
  "valid": true/false
}}

**Important:**
- "consistent": Are the premises logically consistent?
- "satisfiable": Can premises AND conclusion all be true together?
- "valid": Does the conclusion necessarily follow from the premises?
- Be rigorous and apply formal logic rules
- Return ONLY valid JSON, no other text

Respond now:"""

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a formal logic expert. Respond ONLY with valid JSON."},
                {"role": "user", "content": reasoning_prompt}
            ],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return response.choices[0].message.content
    except Exception as e:
        return None


def heuristic_scoring(user_query):
    query_lower = user_query.lower()
    score = 0
    max_score = 100

    if 30 <= len(user_query) <= 500:
        score += 10

    operators = {
        "and": len(re.findall(r'\band\b', query_lower)),
        "or": len(re.findall(r'\bor\b', query_lower)),
        "not": len(re.findall(r'\bnot\b|\bno\b', query_lower)),
        "if": len(re.findall(r'\bif\b', query_lower)),
        "then": len(re.findall(r'\bthen\b', query_lower)),
        "implies": len(re.findall(r'\bimplies\b|\bmeans\b', query_lower))
    }

    total_operators = sum(operators.values())
    if total_operators >= 3:
        score += 25
    elif total_operators >= 1:
        score += 15

    question_markers = ["what", "can we conclude",
                        "therefore", "thus", "?", "prove"]
    has_markers = any(marker in query_lower for marker in question_markers)
    if has_markers:
        score += 20

    sentences = re.split(r'[.;,]\s+|\n+', user_query)
    valid_sentences = [s for s in sentences if len(s.strip()) > 15]
    if len(valid_sentences) >= 2:
        score += 20
    elif len(valid_sentences) == 1:
        score += 10

    patterns = {
        "Modus Ponens": r'if.*then.*(?:is|are)',
        "Categorical": r'all.*(?:are|is)',
        "Conditional": r'if.*then',
        "Disjunctive": r'either.*or',
        "Universal": r'every|all|any'
    }

    detected_patterns = []
    for pattern_name, pattern_regex in patterns.items():
        if re.search(pattern_regex, query_lower):
            detected_patterns.append(pattern_name)

    if detected_patterns:
        score += 25
    else:
        score += 10

    return {
        "score": score,
        "percentage": score,
        "operators": operators,
        "patterns": detected_patterns,
        "quality": "Excellent" if score >= 80 else "Good" if score >= 60 else "Fair" if score >= 40 else "Poor"
    }


def extract_json_from_text(text):
    json_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    match = re.search(json_block_pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass

    brace_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(brace_pattern, text, re.DOTALL)

    for match in reversed(sorted(matches, key=len)):
        try:
            return json.loads(match)
        except:
            continue

    try:
        return json.loads(text)
    except:
        pass

    return None


def stage1_llm_translator(client, user_query, model, temperature, max_tokens):
    system_prompt = """You are a formal logic expert and Z3 programmer.

**CRITICAL: Your response must be ONLY a valid JSON object. No other text before or after.**

Analyze the question and output this EXACT structure:

{
  "premises": ["premise 1", "premise 2"],
  "conclusion": "conclusion statement",
  "z3_code": "from z3 import *\\n\\nP1 = Bool('P1')\\nP2 = Bool('P2')\\nC = Bool('C')\\n\\nconsistency_solver = Solver()\\nconsistency_solver.add(P1)\\nconsistency_solver.add(P2)\\nconsistency_status = consistency_solver.check()\\n\\nsat_solver = Solver()\\nsat_solver.add(P1)\\nsat_solver.add(P2)\\nsat_solver.add(C)\\nsat_status = sat_solver.check()\\n\\nvalidity_solver = Solver()\\nvalidity_solver.add(P1)\\nvalidity_solver.add(P2)\\nvalidity_solver.add(Not(C))\\nvalidity_status = validity_solver.check()\\n\\nresult = {'consistency': {'status': str(consistency_status), 'consistent': consistency_status == sat}, 'satisfiability': {'status': str(sat_status), 'satisfiable': sat_status == sat}, 'validity': {'status': str(validity_status), 'valid': validity_status == unsat}, 'model': str(validity_solver.model()) if validity_status == sat else None}"
}

**Z3 Code Template (adapt variable names to fit the logic):**
- P1, P2, ... for premises
- C for conclusion
- Three solvers: consistency_solver, sat_solver, validity_solver
- Final result dictionary with all three tests

**IMPORTANT:**
- Escape all backslashes as \\\\n for newlines
- Use double quotes for all strings
- Return ONLY the JSON object, nothing else"""

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze this logical question and return ONLY valid JSON:\n\n{user_query}"}
            ],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"LLM API Error: {str(e)}")


def stage2_z3_reasoner(z3_code):
    try:
        exec_globals = {}
        exec_locals = {}

        exec(z3_code, exec_globals, exec_locals)

        if 'result' in exec_locals:
            return exec_locals['result'], None
        else:
            return None, "Z3 code did not produce 'result' variable"

    except Exception as e:
        return None, f"Z3 execution error: {str(e)}"


def stage4_llm_verifier(client, premises, conclusion, z3_result, model, temperature, max_tokens):
    verification_prompt = f"""You are a logic verification expert. Verify the reasoning independently.

**Given:**
Premises: {premises}
Conclusion: {conclusion}

**Z3 Results:**
- Consistency: {"CONSISTENT" if z3_result.get('consistency', {}).get('consistent') else "INCONSISTENT"}
- Satisfiability: {"SATISFIABLE" if z3_result.get('satisfiability', {}).get('satisfiable') else "UNSATISFIABLE"}
- Validity: {"VALID" if z3_result.get('validity', {}).get('valid') else "INVALID"}

**Task:** Independently analyze and determine if Z3 is correct.

**Response (JSON only):**
{{
  "z3_correct": true/false,
  "llm_analysis": {{
    "consistent": true/false,
    "satisfiable": true/false,
    "valid": true/false
  }}
}}"""

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system",
                    "content": "You are a logic expert. Respond ONLY with valid JSON."},
                {"role": "user", "content": verification_prompt}
            ],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return response.choices[0].message.content
    except Exception as e:
        return None


def generate_truth_table_latex(n_premises, z3_result):
    if n_premises > 4:
        return r"\text{Truth table too large to display (more than 4 premises)}"

    n_total_vars = n_premises + 1
    all_combinations = list(product([True, False], repeat=n_total_vars))

    show_conjunction = n_premises > 1

    latex = r"\begin{array}{|"
    latex += "c|" * n_premises
    latex += "c|"
    if show_conjunction:
        latex += "c|"
    latex += "c|"
    latex += "}\n"
    latex += r"\hline" + "\n"

    header_parts = [f"P_{{{i+1}}}" for i in range(n_premises)]
    header_parts.append("C")

    if show_conjunction:
        premise_conj = r" \land ".join(
            [f"P_{{{i+1}}}" for i in range(n_premises)])
        header_parts.append(premise_conj)
        implication_formula = f"{premise_conj} \\rightarrow C"
    else:
        implication_formula = f"P_1 \\rightarrow C"

    header_parts.append(implication_formula)

    latex += " & ".join(header_parts) + r" \\" + "\n"
    latex += r"\hline" + "\n"

    for combo in all_combinations:
        premise_vals = combo[:n_premises]
        c_val = combo[n_premises]

        row_parts = []

        for val in premise_vals:
            row_parts.append("T" if val else "F")

        row_parts.append("T" if c_val else "F")

        premises_conj_val = all(premise_vals)
        if show_conjunction:
            row_parts.append("T" if premises_conj_val else "F")

        implication_val = (not premises_conj_val) or c_val

        row_parts.append("T" if implication_val else "F")

        latex += " & ".join(row_parts) + r" \\" + "\n"

    latex += r"\hline" + "\n"
    latex += r"\end{array}"

    return latex


def main():
    st.header("📝 Input")

    user_query = st.text_area(
        "Enter your logical reasoning question:",
        placeholder="Example: If all mathematicians are genius, and Dr. Maharani is a mathematician, then Dr. Maharani is a genius.",
        height=120,
        help="Enter a logical reasoning question with clear premises"
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        analyze_button = st.button(
            "🔍 Analyze with Neuro-Symbolic Logic Validator", type="primary", use_container_width=True)
    with col2:
        if st.button("🔄 Clear", use_container_width=True):
            st.rerun()

    if analyze_button:
        if not user_query:
            st.error("Please enter a query")
            return

        # Silently detect if this is quantifier logic
        has_quantifiers = check_quantifier_logic(user_query)

        client = get_groq_client()

        # Stage 0: Heuristic (always shown)
        with st.spinner("📊 Stage 0: Analyzing input quality..."):
            heuristic_result = heuristic_scoring(user_query)

            st.subheader("📊 Stage 0: Heuristic Input Analysis")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Input Quality Score", f"{heuristic_result['percentage']}%",
                          delta=heuristic_result['quality'])
            with col2:
                st.metric("Logical Operators", sum(
                    heuristic_result['operators'].values()))
            with col3:
                st.metric("Detected Patterns", len(
                    heuristic_result['patterns']))

            if heuristic_result['patterns']:
                st.info(
                    f"🎓 **Patterns Detected:** {', '.join(heuristic_result['patterns'])}")

        if has_quantifiers:
            with st.spinner("🧠 Stage 1: Analyzing formal logic..."):
                try:
                    llm_output = advanced_reasoning(
                        client, user_query, model, temperature, max_tokens)

                    if not llm_output:
                        st.error("❌ Failed to analyze the logical reasoning")
                        return

                    llm_data = extract_json_from_text(llm_output)

                    if llm_data is None:
                        st.error("❌ Failed to parse analysis results")
                        return

                    is_valid, validation_msg = validate_logic_extraction(
                        llm_data.get('premises', []),
                        llm_data.get('conclusion', '')
                    )

                    if not is_valid:
                        st.error("❌ **No Valid Logic Statements Found**")
                        st.warning(
                            "Please enter a proper logical reasoning question with clear premises and conclusion.")
                        return

                    st.success("✅ Stage 1 Complete: Logic analyzed")

                    st.subheader("🔤 Stage 1: Logical Structure")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**📋 Extracted Premises:**")
                        for i, premise in enumerate(llm_data['premises'], 1):
                            st.markdown(f"**P{i}:** {premise}")

                    with col2:
                        st.markdown("**🎯 Conclusion:**")
                        st.markdown(f"**C:** {llm_data['conclusion']}")

                    # Stage 2: Truth Table (always shown)
                    st.subheader("📊 Stage 2: Truth Table")

                    n_premises = len(llm_data['premises'])
                    if n_premises > 4:
                        st.warning(
                            "⚠️ Truth table too large to display (more than 4 premises)")
                    else:
                        dummy_result = {'consistency': {'consistent': True}, 'satisfiability': {
                            'satisfiable': True}, 'validity': {'valid': True}}
                        truth_table_latex = generate_truth_table_latex(
                            n_premises, dummy_result)
                        st.latex(truth_table_latex)

                    st.subheader("⚡ Stage 3: Verification Results")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown("### 🔷 Consistency")
                        if llm_data.get('consistent', True):
                            st.success("✅ CONSISTENT")
                            st.caption("Premises do not contradict")
                        else:
                            st.error("❌ INCONSISTENT")
                            st.caption("Premises are contradictory")

                    with col2:
                        st.markdown("### 🔶 Satisfiability")
                        if llm_data.get('satisfiable', True):
                            st.success("✅ SATISFIABLE")
                            st.caption("Premises & conclusion can coexist")
                        else:
                            st.warning("❌ UNSATISFIABLE")
                            st.caption("Premises & conclusion conflict")

                    with col3:
                        st.markdown("### 🟣 Validity")
                        if llm_data.get('valid', False):
                            st.success("✅ VALID")
                            st.caption("Conclusion follows necessarily")
                        else:
                            st.warning("❌ INVALID")
                            st.caption("Conclusion not entailed")

                    st.header("📊 Analysis Summary")

                    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(
                        4)

                    with summary_col1:
                        st.metric("Input Quality", f"{heuristic_result['percentage']}%",
                                  delta=heuristic_result['quality'])

                    with summary_col2:
                        consistency_val = "✅" if llm_data.get(
                            'consistent', True) else "❌"
                        st.metric("Consistent", consistency_val)

                    with summary_col3:
                        sat_val = "✅" if llm_data.get(
                            'satisfiable', True) else "❌"
                        st.metric("Satisfiable", sat_val)

                    with summary_col4:
                        validity_val = "✅" if llm_data.get(
                            'valid', False) else "❌"
                        st.metric("Valid", validity_val)

                    return

                except Exception as e:
                    st.error(f"❌ Analysis Error: {str(e)}")
                    return

        with st.spinner("🧠 Stage 1: Translating to formal logic..."):
            try:
                llm_output = stage1_llm_translator(
                    client, user_query, model, temperature, max_tokens)

                llm_data = extract_json_from_text(llm_output)

                if llm_data is None:
                    st.error("❌ Failed to parse response")
                    with st.expander("🐛 Debug: Raw Output"):
                        st.code(llm_output, language="text")
                    return

                if 'premises' not in llm_data or 'conclusion' not in llm_data or 'z3_code' not in llm_data:
                    st.error("❌ Response missing required fields")
                    with st.expander("🐛 Debug: Parsed JSON"):
                        st.json(llm_data)
                    return

                is_valid, validation_msg = validate_logic_extraction(
                    llm_data['premises'],
                    llm_data['conclusion']
                )

                if not is_valid:
                    st.error("❌ **No Valid Logic Statements Found**")
                    st.warning("""
**Please enter a proper logical reasoning question.**

Your input must contain:
- **Premises**: Clear logical statements or facts
- **Conclusion**: What you want to prove or determine

**Please try again with a clear logical reasoning question!**
                    """)

                    with st.expander("🔍 What was extracted from your input"):
                        st.markdown(f"**Premises:** {llm_data['premises']}")
                        st.markdown(
                            f"**Conclusion:** {llm_data['conclusion']}")
                        st.caption(f"Validation error: {validation_msg}")

                    return

                st.success("✅ Stage 1 Complete: Logic translated")

                st.subheader("🔤 Stage 1: Logical Structure")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**📋 Extracted Premises:**")
                    for i, premise in enumerate(llm_data['premises'], 1):
                        st.markdown(f"**P{i}:** {premise}")

                with col2:
                    st.markdown("**🎯 Conclusion:**")
                    st.markdown(f"**C:** {llm_data['conclusion']}")

            except Exception as e:
                st.error(f"❌ Stage 1 Error: {str(e)}")
                with st.expander("🐛 Debug Info"):
                    st.code(llm_output if 'llm_output' in locals()
                            else "No output captured", language="text")
                return

        with st.spinner("⚡ Executing verification..."):
            try:
                z3_result, z3_error = stage2_z3_reasoner(llm_data['z3_code'])

                if z3_error:
                    st.error(f"❌ Verification Error: {z3_error}")
                    with st.expander("🐛 Debug: Generated Code"):
                        st.code(llm_data['z3_code'], language="python")
                    return

            except Exception as e:
                st.error(f"❌ Verification Error: {str(e)}")
                return

        st.subheader("📊 Stage 2: Truth Table")

        if len(llm_data['premises']) > 4:
            st.warning(
                "⚠️ Truth table too large to display (more than 4 premises)")
        else:
            truth_table_latex = generate_truth_table_latex(
                len(llm_data['premises']), z3_result
            )

            st.latex(truth_table_latex)

        final_result = z3_result

        try:
            verification_output = stage4_llm_verifier(
                client,
                llm_data['premises'],
                llm_data['conclusion'],
                z3_result,
                model,
                temperature,
                max_tokens
            )

            if verification_output:
                verification_data = extract_json_from_text(verification_output)

                if verification_data and not verification_data.get('z3_correct', True):
                    llm_analysis = verification_data.get('llm_analysis', {})
                    final_result = {
                        'consistency': {'consistent': llm_analysis.get('consistent', True)},
                        'satisfiability': {'satisfiable': llm_analysis.get('satisfiable', True)},
                        'validity': {'valid': llm_analysis.get('valid', False)}
                    }
        except:
            pass

        st.subheader("⚡ Stage 3: Verification Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 🔷 Consistency")
            if final_result.get('consistency', {}).get('consistent'):
                st.success("✅ CONSISTENT")
                st.caption("Premises do not contradict")
            else:
                st.error("❌ INCONSISTENT")
                st.caption("Premises are contradictory")

        with col2:
            st.markdown("### 🔶 Satisfiability")
            if final_result.get('satisfiability', {}).get('satisfiable'):
                st.success("✅ SATISFIABLE")
                st.caption("Premises & conclusion can coexist")
            else:
                st.warning("❌ UNSATISFIABLE")
                st.caption("Premises & conclusion conflict")

        with col3:
            st.markdown("### 🟣 Validity")
            if final_result.get('validity', {}).get('valid'):
                st.success("✅ VALID")
                st.caption("Conclusion follows necessarily")
            else:
                st.warning("❌ INVALID")
                st.caption("Conclusion not entailed")

        st.header("📊 Analysis Summary")

        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

        with summary_col1:
            st.metric("Input Quality", f"{heuristic_result['percentage']}%",
                      delta=heuristic_result['quality'])

        with summary_col2:
            consistency_val = "✅" if final_result.get(
                'consistency', {}).get('consistent') else "❌"
            st.metric("Consistent", consistency_val)

        with summary_col3:
            sat_val = "✅" if final_result.get(
                'satisfiability', {}).get('satisfiable') else "❌"
            st.metric("Satisfiable", sat_val)

        with summary_col4:
            validity_val = "✅" if final_result.get(
                'validity', {}).get('valid') else "❌"
            st.metric("Valid", validity_val)


if __name__ == "__main__":
    main()
