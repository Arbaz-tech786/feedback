import streamlit as st
import json
import boto3
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import logging
from decimal import Decimal
from aws_config_fix import init_aws_clients_with_retry, check_aws_resources_exist, get_resource_status_message

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize AWS clients with improved error handling
@st.cache_resource
def init_aws_clients():
    """Initialize AWS clients with caching and improved error handling"""
    try:
        dynamodb, s3, bedrock_runtime = init_aws_clients_with_retry()
        
        if not dynamodb or not s3 or not bedrock_runtime:
            st.error("❌ Failed to initialize AWS clients. Please check your AWS configuration.")
            return None, None, None
        
        # Check if required resources exist
        table_names = [IDEAS_TABLE, FINALIZE_REPORT_TABLE, SOCIAL_LENS_TABLE,
                      FORM_RESPONSES_TABLE, MEETING_SUMMARY_TABLE, FEEDBACK_TABLE]
        resource_status = check_aws_resources_exist(dynamodb, s3, bedrock_runtime, table_names, MATCH_MAKING_BUCKET)
        
        if not resource_status['all_resources_available']:
            status_message = get_resource_status_message(resource_status)
            st.error(f"⚠️ AWS Resource Access Issues:\n{status_message}")
            st.info("💡 Please ensure all required AWS resources exist and your credentials have proper permissions.")
            
            # Still return clients so the app can function with partial resources
            return dynamodb, s3, bedrock_runtime
        
        return dynamodb, s3, bedrock_runtime
        
    except Exception as e:
        st.error(f"❌ Failed to initialize AWS clients: {str(e)}")
        logger.error(f"AWS client initialization error: {str(e)}")
        return None, None, None

# Configuration - Load from Streamlit secrets or environment variables
def get_config(key: str, default: str) -> str:
    """Get configuration from Streamlit secrets or environment variables"""
    try:
        if hasattr(st, 'secrets') and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key, default)

IDEAS_TABLE = get_config('IDEAS_TABLE', 'ideas')
FINALIZE_REPORT_TABLE = get_config('FINALIZE_REPORT_TABLE', 'finalize_report')
SOCIAL_LENS_TABLE = get_config('SOCIAL_LENS_TABLE', 'Social_Lens')
MATCH_MAKING_BUCKET = get_config('MATCH_MAKING_BUCKET', 'outlawml-testing')
FORM_RESPONSES_TABLE = get_config('FORM_RESPONSES_TABLE', 'form_responses')
MEETING_SUMMARY_TABLE = get_config('MEETING_SUMMARY_TABLE', 'meeting_summary')
FEEDBACK_TABLE = get_config('FEEDBACK_TABLE', 'feedback')
BEDROCK_MODEL_ID = get_config('BEDROCK_MODEL_ID', 'anthropic.claude-3-sonnet-20240229-v1:0')

# Initialize session state
if 'questions' not in st.session_state:
    st.session_state.questions = []
if 'responses' not in st.session_state:
    st.session_state.responses = {}
if 'current_stage' not in st.session_state:
    st.session_state.current_stage = None
if 'ai_request_id' not in st.session_state:
    st.session_state.ai_request_id = None
if 'analysis_summary' not in st.session_state:
    st.session_state.analysis_summary = ''

def invoke_bedrock(prompt: str, max_tokens: int = 2000) -> str:
    """Invoke Bedrock model with the given prompt"""
    try:
        dynamodb, s3, bedrock_runtime = init_aws_clients()
        if not bedrock_runtime:
            raise Exception("Bedrock client not initialized")
        
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        })
        
        response = bedrock_runtime.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            body=body
        )
        
        response_body = json.loads(response.get('body').read())
        return response_body.get('content')[0].get('text')
    
    except Exception as e:
        logger.error(f"Error invoking Bedrock: {str(e)}")
        st.error(f"Error invoking Bedrock: {str(e)}")
        return None

def extract_domain(input_data: Dict) -> str:
    """Extract domain from input data"""
    tags = input_data.get('tags', [])
    for tag in tags:
        tag_lower = tag.lower()
        if 'healthcare' in tag_lower or 'medical' in tag_lower:
            return 'healthcare'
        elif 'fintech' in tag_lower or 'financial' in tag_lower:
            return 'fintech'
        elif 'e-commerce' in tag_lower or 'ecommerce' in tag_lower:
            return 'e-commerce'
        elif 'b2b' in tag_lower or 'saas' in tag_lower:
            return 'b2b-saas'
        elif 'hardware' in tag_lower:
            return 'hardware'

    description = input_data.get('description', '').lower()
    if 'healthcare' in description or 'medical' in description:
        return 'healthcare'
    elif 'fintech' in description or 'financial' in description:
        return 'fintech'
    elif 'e-commerce' in description or 'ecommerce' in description:
        return 'e-commerce'
    elif 'b2b' in description or 'saas' in description:
        return 'b2b-saas'
    elif 'hardware' in description:
        return 'hardware'

    return 'unknown'

def generate_dynamic_options(agent_name: str, context_summary: str, question_focus: str) -> List[Dict]:
    """
    Generate contextually relevant multiple-choice options using AI.

    Args:
        agent_name: Name of the AI agent being evaluated
        context_summary: Brief summary of the agent's output/context
        question_focus: What aspect is being evaluated (e.g., 'accuracy', 'relevance', 'quality')

    Returns:
        List of option dictionaries with 'value' and 'label' keys
    """
    prompt = f"""You are an expert in creating effective evaluation scales for AI agent performance assessment.

Task: Generate 3-4 multiple-choice options for evaluating {agent_name}'s performance on: {question_focus}

Context: {context_summary}

Requirements:
1. Create 3-4 distinct options that cover the full quality spectrum (excellent to poor)
2. Make labels SHORT (2-4 words max), clear, and user-friendly
3. Ensure options are mutually exclusive and comprehensive
4. Use natural language that founders/users would understand
5. Values should be snake_case identifiers

Output Format (STRICT JSON):
{{
  "options": [
    {{"value": "excellent", "label": "Spot on"}},
    {{"value": "good", "label": "Mostly accurate"}},
    {{"value": "fair", "label": "Partially helpful"}},
    {{"value": "poor", "label": "Missed the mark"}}
  ]
}}

Examples of good option sets:

For accuracy evaluation:
- {{"value": "perfect", "label": "Perfect accuracy"}}
- {{"value": "good", "label": "Mostly accurate"}}
- {{"value": "partial", "label": "Partially accurate"}}
- {{"value": "inaccurate", "label": "Not accurate"}}

For relevance evaluation:
- {{"value": "highly_relevant", "label": "Highly relevant"}}
- {{"value": "relevant", "label": "Relevant"}}
- {{"value": "somewhat", "label": "Somewhat relevant"}}
- {{"value": "not_relevant", "label": "Not relevant"}}

For priority/ordering evaluation:
- {{"value": "perfect_order", "label": "Perfect priority"}}
- {{"value": "good_order", "label": "Good priority"}}
- {{"value": "wrong_order", "label": "Wrong priority"}}
- {{"value": "unsure", "label": "Not sure"}}

Generate contextually appropriate options now (JSON only):"""

    try:
        response = invoke_bedrock(prompt, max_tokens=500)
        if response:
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                options_data = json.loads(json_match.group())
                options = options_data.get('options', [])

                # Validate options
                if len(options) >= 3 and all('value' in opt and 'label' in opt for opt in options):
                    return options[:4]  # Limit to 4 options max

        # Fallback options if generation fails
        logger.warning(f"Failed to generate options for {agent_name}, using fallback")
        return get_fallback_options(question_focus)

    except Exception as e:
        logger.error(f"Error generating dynamic options: {str(e)}")
        return get_fallback_options(question_focus)

def get_fallback_options(question_focus: str) -> List[Dict]:
    """Provide fallback options based on question focus_area"""
    fallback_map = {
        # Idea Capture focus areas
        'issue_identification_accuracy': [
            {"value": "root_causes", "label": "Root causes"},
            {"value": "mostly_root", "label": "Mostly root"},
            {"value": "surface_level", "label": "Surface-level"},
            {"value": "symptoms_only", "label": "Just symptoms"}
        ],
        'hypothesis_quality': [
            {"value": "excellent", "label": "Excellent"},
            {"value": "good", "label": "Good"},
            {"value": "fair", "label": "Fair"},
            {"value": "poor", "label": "Poor"}
        ],
        'insight_depth': [
            {"value": "deep", "label": "Deep insights"},
            {"value": "good", "label": "Good depth"},
            {"value": "shallow", "label": "Shallow"},
            {"value": "surface", "label": "Surface-level"}
        ],

        # Lens Selector focus areas
        'lens_priority_correctness': [
            {"value": "perfect_priority", "label": "Perfect priority"},
            {"value": "good_priority", "label": "Good priority"},
            {"value": "wrong_priority", "label": "Wrong priority"},
            {"value": "unsure", "label": "Not sure"}
        ],
        'sequencing_logic': [
            {"value": "excellent_sequence", "label": "Excellent sequence"},
            {"value": "good_sequence", "label": "Good sequence"},
            {"value": "poor_sequence", "label": "Poor sequence"},
            {"value": "needs_reorder", "label": "Needs reorder"}
        ],

        # Survey Generator focus areas
        'hypothesis_testing_quality': [
            {"value": "tests_well", "label": "Tests hypotheses"},
            {"value": "mostly_tests", "label": "Mostly tests"},
            {"value": "generic", "label": "Generic questions"},
            {"value": "doesnt_test", "label": "Doesn't test"}
        ],
        'question_clarity': [
            {"value": "very_clear", "label": "Very clear"},
            {"value": "clear", "label": "Clear"},
            {"value": "somewhat_clear", "label": "Somewhat clear"},
            {"value": "unclear", "label": "Unclear"}
        ],

        # 360 Report focus areas
        'verdict_accuracy': [
            {"value": "fully_aligned", "label": "Fully aligned"},
            {"value": "mostly_aligned", "label": "Mostly aligned"},
            {"value": "partly_aligned", "label": "Partly aligned"},
            {"value": "misaligned", "label": "Misaligned"}
        ],
        'strategic_alignment': [
            {"value": "matches_view", "label": "Matches my view"},
            {"value": "partly_matches", "label": "Partly matches"},
            {"value": "conflicts", "label": "Conflicts"},
            {"value": "totally_wrong", "label": "Totally wrong"}
        ],

        # Social Lens focus areas
        'buzz_score_accuracy': [
            {"value": "accurate", "label": "Accurate"},
            {"value": "slightly_high", "label": "Bit high"},
            {"value": "slightly_low", "label": "Bit low"},
            {"value": "way_off", "label": "Way off"}
        ],
        'source_quality': [
            {"value": "excellent", "label": "Excellent sources"},
            {"value": "good", "label": "Good sources"},
            {"value": "fair", "label": "Fair sources"},
            {"value": "poor", "label": "Poor sources"}
        ],

        # Match Maker focus areas
        'match_relevance': [
            {"value": "perfect_match", "label": "Perfect match"},
            {"value": "good_match", "label": "Good match"},
            {"value": "fair_match", "label": "Fair match"},
            {"value": "poor_match", "label": "Poor match"}
        ],
        'expertise_fit': [
            {"value": "direct_fit", "label": "Direct fit"},
            {"value": "good_overlap", "label": "Good overlap"},
            {"value": "partial_fit", "label": "Partial fit"},
            {"value": "weak_fit", "label": "Weak fit"}
        ],

        # Survey Analysis focus areas
        'segment_accuracy': [
            {"value": "matches_reality", "label": "Matches reality"},
            {"value": "mostly_accurate", "label": "Mostly accurate"},
            {"value": "some_gaps", "label": "Some gaps"},
            {"value": "way_off", "label": "Way off"}
        ],
        'pain_point_identification': [
            {"value": "real_pains", "label": "Real pain points"},
            {"value": "mostly_real", "label": "Mostly real"},
            {"value": "some_accuracy", "label": "Some accuracy"},
            {"value": "missed_reality", "label": "Missed reality"}
        ],
        'opportunity_quality': [
            {"value": "actionable_opps", "label": "Actionable"},
            {"value": "mostly_actionable", "label": "Mostly actionable"},
            {"value": "vague_opps", "label": "Vague"},
            {"value": "not_actionable", "label": "Not actionable"}
        ],
        'sentiment_calibration': [
            {"value": "accurate_sentiment", "label": "Accurate"},
            {"value": "mostly_accurate", "label": "Mostly accurate"},
            {"value": "somewhat_off", "label": "Somewhat off"},
            {"value": "way_off", "label": "Way off"}
        ],
        'strategic_clarity': [
            {"value": "very_clear", "label": "Very clear"},
            {"value": "clear", "label": "Clear"},
            {"value": "somewhat_clear", "label": "Somewhat clear"},
            {"value": "unclear", "label": "Unclear"}
        ],

        # Meeting Summary focus areas
        'prep_relevance': [
            {"value": "right_on_target", "label": "Right on target"},
            {"value": "mostly_helpful", "label": "Mostly helpful"},
            {"value": "some_value", "label": "Some value"},
            {"value": "off_target", "label": "Off target"}
        ],
        'summary_completeness': [
            {"value": "comprehensive", "label": "Comprehensive"},
            {"value": "good_coverage", "label": "Good coverage"},
            {"value": "missing_some", "label": "Missing some"},
            {"value": "incomplete", "label": "Incomplete"}
        ],
        'actionability': [
            {"value": "highly_actionable", "label": "Highly actionable"},
            {"value": "actionable", "label": "Actionable"},
            {"value": "somewhat_actionable", "label": "Somewhat"},
            {"value": "not_actionable", "label": "Not actionable"}
        ],
        'clarity': [
            {"value": "crystal_clear", "label": "Crystal clear"},
            {"value": "clear", "label": "Clear"},
            {"value": "somewhat_clear", "label": "Somewhat clear"},
            {"value": "vague", "label": "Vague"}
        ],
        'prioritization': [
            {"value": "perfect_priority", "label": "Perfect priority"},
            {"value": "good_priority", "label": "Good priority"},
            {"value": "poor_priority", "label": "Poor priority"},
            {"value": "no_priority", "label": "No priority"}
        ],
        'time_efficiency': [
            {"value": "very_efficient", "label": "Very efficient"},
            {"value": "efficient", "label": "Efficient"},
            {"value": "somewhat_efficient", "label": "Somewhat"},
            {"value": "time_waster", "label": "Time waster"}
        ],

        # Transcript Summary focus areas
        'insight_accuracy': [
            {"value": "spot_on", "label": "Spot on"},
            {"value": "mostly_right", "label": "Mostly right"},
            {"value": "partially_right", "label": "Partially right"},
            {"value": "missed_key_points", "label": "Missed key points"}
        ],
        'confidence_calibration': [
            {"value": "well_calibrated", "label": "Well calibrated"},
            {"value": "mostly_calibrated", "label": "Mostly calibrated"},
            {"value": "overconfident", "label": "Overconfident"},
            {"value": "underconfident", "label": "Underconfident"}
        ],
        'impact_assessment': [
            {"value": "correct_impact", "label": "Correct impact"},
            {"value": "mostly_correct", "label": "Mostly correct"},
            {"value": "some_misses", "label": "Some misses"},
            {"value": "wrong_priorities", "label": "Wrong priorities"}
        ],
        'quote_relevance': [
            {"value": "perfect_quotes", "label": "Perfect quotes"},
            {"value": "good_quotes", "label": "Good quotes"},
            {"value": "fair_quotes", "label": "Fair quotes"},
            {"value": "poor_quotes", "label": "Poor quotes"}
        ],
        'completeness': [
            {"value": "got_everything", "label": "Got everything"},
            {"value": "got_most", "label": "Got most"},
            {"value": "missed_some", "label": "Missed some"},
            {"value": "missed_a_lot", "label": "Missed a lot"}
        ],
        'persona_detection': [
            {"value": "correct_persona", "label": "Correct persona"},
            {"value": "close_enough", "label": "Close enough"},
            {"value": "somewhat_off", "label": "Somewhat off"},
            {"value": "wrong_persona", "label": "Wrong persona"}
        ],

        # Generic fallbacks
        'accuracy': [
            {"value": "excellent", "label": "Spot on"},
            {"value": "good", "label": "Mostly right"},
            {"value": "fair", "label": "Partially right"},
            {"value": "poor", "label": "Missed the mark"}
        ],
        'relevance': [
            {"value": "highly_relevant", "label": "Highly relevant"},
            {"value": "relevant", "label": "Relevant"},
            {"value": "somewhat", "label": "Somewhat"},
            {"value": "not_relevant", "label": "Not relevant"}
        ],
        'quality': [
            {"value": "excellent", "label": "Excellent"},
            {"value": "good", "label": "Good"},
            {"value": "fair", "label": "Fair"},
            {"value": "poor", "label": "Poor"}
        ]
    }

    # Return the most appropriate fallback or default
    return fallback_map.get(question_focus.lower(), fallback_map['quality'])

def build_idea_capture_prompt(burning_issues: List[Dict], domain: str, input_metadata: Dict) -> str:
    """Build the enhanced prompt for the Idea Capture agent with dynamic option generation"""
    stage = input_metadata.get('stage', 'unknown')

    # Build the burning issues text with rich context
    issues_text = ""
    for i, issue in enumerate(burning_issues, 1):
        title = issue.get('title', '')
        hypothesis = issue.get('hypothesis', '')
        confidence = 0.3 if 'undefined' in title.lower() or 'zero' in title.lower() else 0.6
        issues_text += f"""
  Issue #{i}: {title}
    → Hypothesis: {hypothesis}
    → Confidence: {confidence:.1f}
"""

    prompt = f"""You are an expert Meta-Feedback Generator specializing in evaluating AI agent performance in startup validation contexts.

═══════════════════════════════════════════════════════════════════════════════
AGENT BEING EVALUATED: Idea Capture Agent
DOMAIN: {domain.upper()}
STARTUP STAGE: {stage}
═══════════════════════════════════════════════════════════════════════════════

CONTEXT & AGENT OUTPUT:
The Idea Capture agent analyzed the founder's input and identified these burning issues:
{issues_text}

═══════════════════════════════════════════════════════════════════════════════
YOUR TASK:
Generate 1-2 concise evaluation questions that assess how effectively the AI identified the founder's core challenges.

EVALUATION CRITERIA TO CONSIDER:
1. **Accuracy**: Did the AI correctly identify the real business challenges?
2. **Relevance**: Are these issues actually critical for this {domain} startup at {stage} stage?
3. **Insight Quality**: Do the hypotheses demonstrate deep understanding or surface-level analysis?
4. **Completeness**: Are key challenges missing? Are irrelevant issues included?
5. **Actionability**: Can the founder act on these insights?

EXAMPLES OF EXCELLENT EVALUATION QUESTIONS:
✓ "How accurately did we identify your top business challenges?" (focuses on core accuracy)
✓ "Do these burning issues reflect your actual situation?" (tests relevance)
✓ "Quality of our problem hypotheses?" (assesses insight depth)

EXAMPLES OF POOR EVALUATION QUESTIONS:
✗ "What are your main challenges?" (data collection, not evaluation)
✗ "How do you feel about your business?" (too vague, not AI-performance focused)
✗ "Did you enjoy using our AI agent for identifying challenges and problems?" (not evaluating output quality)

CHAIN-OF-THOUGHT REASONING PROCESS:
Before generating questions, consider:
1. What aspect of the AI's output is most critical to evaluate?
2. What would a founder need to know to judge if the AI helped them?
3. How can we capture this in under 120 characters?

ANALYSIS & OUTPUT FORMAT:

STEP 1: Analyze the burning issues quality
Before generating questions, analyze:
- Are these ROOT CAUSES or just symptoms?
- Do hypotheses have clear, measurable success thresholds?
- Is confidence calibration appropriate (not overconfident)?
- Are issues actually BURNING (critical) or nice-to-have?
- Does the AI demonstrate deep domain understanding?

STEP 2: Generate output in this EXACT JSON format:
{{
  "questions": [
    {{
      "id": "q1",
      "text": "Specific, actionable evaluation question (max 150 chars)",
      "type": "multiple_choice",
      "focus_area": "issue_identification_accuracy|hypothesis_quality|insight_depth|problem_relevance|confidence_calibration",
      "options": [
        {{"value": "excellent_perf", "label": "Context-specific label for EXCELLENT performance"}},
        {{"value": "good_perf", "label": "Context-specific label for GOOD performance"}},
        {{"value": "fair_perf", "label": "Context-specific label for FAIR performance"}},
        {{"value": "poor_perf", "label": "Context-specific label for POOR performance"}}
      ],
      "rl_value": 0.85
    }},
    {{
      "id": "q2",
      "text": "Optional second question (max 150 chars)",
      "type": "multiple_choice",
      "focus_area": "issue_identification_accuracy|hypothesis_quality|insight_depth|problem_relevance|confidence_calibration",
      "options": [
        {{"value": "excellent_perf", "label": "Context-specific label"}},
        {{"value": "good_perf", "label": "Context-specific label"}},
        {{"value": "fair_perf", "label": "Context-specific label"}},
        {{"value": "poor_perf", "label": "Context-specific label"}}
      ],
      "rl_value": 0.80
    }}
  ],
  "analysis_summary": "2-3 sentence summary analyzing the quality of the burning issues. Example: 'The AI identified 3 specific, testable issues with clear hypotheses. BI #1 addresses root cause (market segmentation) while BI #2-3 focus on differentiation. Confidence levels (0.6-0.8) are appropriately calibrated. Overall: Strong identification with good depth.'"
}}

IMPORTANT CONSTRAINTS:
- MUST generate EXACTLY 2 questions (not 1, not more than 2)
- Each question MUST be under 150 characters
- Use "focus_area" (not "focus") with specific names like "hypothesis_quality" not just "quality"
- Generate OPTIONS directly in the response (context-specific, not generic)
- Options must reflect ACTUAL performance shown in your analysis
- Include "analysis_summary" explaining what makes this output high/low quality
- RL values should reflect question importance (0.7-0.95 range)
- Return ONLY valid JSON, no additional text

OPTION GENERATION RULES:
Make option labels SPECIFIC to the actual performance, not generic:
❌ BAD (generic): ["Spot on", "Mostly right", "Wrong", "Very wrong"]
✅ GOOD (contextual): ["Root causes", "Mostly root", "Surface-level", "Just symptoms"]

Generate the evaluation questions with analysis now:"""

    return prompt

def build_lens_selector_prompt(results: List[Dict], domain: str, burning_issues_summary: List[str]) -> str:
    """Build the enhanced prompt for the Lens Selector agent"""
    lens_order = [result['lens'] for result in results]
    first_lens = lens_order[0] if lens_order else None
    confidence_1 = results[0].get('confidence', 0.5) if results else 0.5

    # Build rich lens order text with confidence scores
    lens_text = ""
    for i, result in enumerate(results[:4], 1):
        lens_name = result.get('lens', 'Unknown')
        confidence = result.get('confidence', 0.0)
        rationale = result.get('rationale', '')[:100]  # First 100 chars
        lens_text += f"  {i}. {lens_name} (confidence: {confidence:.2f})\n"
        if rationale:
            lens_text += f"     Rationale: {rationale}...\n"

    # Build burning issues context
    issues_context = "\n".join([f"  • {issue}" for issue in burning_issues_summary[:3]])

    prompt = f"""You are an expert Meta-Feedback Generator specializing in validation methodology assessment for startups.

═══════════════════════════════════════════════════════════════════════════════
AGENT BEING EVALUATED: Lens Selector Agent
DOMAIN: {domain.upper()}
PRIMARY CHALLENGE AREAS:
{issues_context}
═══════════════════════════════════════════════════════════════════════════════

CONTEXT & AGENT OUTPUT:
The Lens Selector agent analyzed the founder's burning issues and recommended this prioritization for validation lenses:

RECOMMENDED VALIDATION ORDER:
{lens_text}

PRIMARY RECOMMENDATION: {first_lens}
CONFIDENCE LEVEL: {confidence_1:.0%}

═══════════════════════════════════════════════════════════════════════════════
YOUR TASK:
Generate 1-2 sharp evaluation questions that assess whether the AI correctly prioritized the validation approach.

EVALUATION CRITERIA TO CONSIDER:
1. **Priority Correctness**: Is the #1 recommended lens actually the best starting point?
2. **Sequencing Logic**: Does the order make strategic sense for this {domain} business?
3. **Context Alignment**: Do recommendations match the burning issues identified?
4. **Confidence Calibration**: Is the AI appropriately confident/uncertain?
5. **Domain Expertise**: Does the prioritization show understanding of {domain} validation needs?

WHY LENS PRIORITIZATION MATTERS:
Founders have limited time/resources. Starting with the wrong validation lens can:
- Waste months pursuing irrelevant data
- Miss critical risk factors
- Lead to false confidence or unnecessary pivots

EXAMPLES OF EXCELLENT EVALUATION QUESTIONS:
✓ "Is {first_lens} the right validation lens to start with?" (direct priority test)
✓ "Does this lens order match your validation needs?" (strategic alignment)
✓ "Quality of our lens prioritization for {domain}?" (domain-specific assessment)

EXAMPLES OF POOR EVALUATION QUESTIONS:
✗ "Which lens should we use?" (asking user to do the AI's job)
✗ "Do you like our recommendations?" (too vague, not actionable)
✗ "How many lenses should we include in the analysis process?" (wrong focus)

CHAIN-OF-THOUGHT REASONING:
Before generating questions, think:
1. What's the highest-risk failure mode? (wrong first lens = wasted time)
2. Can the founder confidently judge if this order is right?
3. Should we ask about the #1 lens or the overall sequence?

ANALYSIS & OUTPUT FORMAT:

STEP 1: Analyze the lens prioritization quality
Before generating questions, analyze:
- Is the #1 lens appropriate given the domain and stage?
- Does the sequence make strategic sense (e.g., validate market before building)?
- Are confidence scores well-calibrated?
- Do pros/cons show deep vs shallow reasoning?
- Would a different order save time/money?

STEP 2: Generate output in this EXACT JSON format:
{{
  "questions": [
    {{
      "id": "q1",
      "text": "Specific, actionable evaluation question (max 150 chars)",
      "type": "multiple_choice",
      "focus_area": "lens_priority_correctness|sequencing_logic|strategic_fit|confidence_calibration",
      "options": [
        {{"value": "perfect_priority", "label": "Context-specific label for PERFECT prioritization"}},
        {{"value": "good_priority", "label": "Context-specific label for GOOD prioritization"}},
        {{"value": "wrong_priority", "label": "Context-specific label for WRONG prioritization"}},
        {{"value": "unsure", "label": "Context-specific label for UNSURE"}}
      ],
      "rl_value": 0.88
    }},
    {{
      "id": "q2",
      "text": "Optional second question (max 150 chars)",
      "type": "multiple_choice",
      "focus_area": "lens_priority_correctness|sequencing_logic|strategic_fit|confidence_calibration",
      "options": [
        {{"value": "excellent", "label": "Context-specific label"}},
        {{"value": "good", "label": "Context-specific label"}},
        {{"value": "fair", "label": "Context-specific label"}},
        {{"value": "poor", "label": "Context-specific label"}}
      ],
      "rl_value": 0.82
    }}
  ],
  "analysis_summary": "2-3 sentence summary analyzing the lens prioritization quality. Example: 'Survey is correctly prioritized as #1 for an ideation-stage {domain} product with undefined ICP. Confidence (0.8) is appropriate. However, SME is ranked #3 but could be #2 given technical validation needs. Sequencing logic is mostly sound but could be optimized.'"
}}

IMPORTANT CONSTRAINTS:
- MUST generate EXACTLY 2 questions (not 1, not more than 2)
- Each question MUST be under 150 characters
- Use "focus_area" (not "focus") with specific names like "lens_priority_correctness"
- Generate OPTIONS directly in the response (context-specific to the actual prioritization)
- Options should reflect whether prioritization/sequence is correct
- Include "analysis_summary" explaining quality of lens prioritization
- RL values should reflect strategic importance (0.75-0.92 range)
- Return ONLY valid JSON

OPTION GENERATION RULES:
Make option labels SPECIFIC to prioritization quality:
❌ BAD (generic): ["Perfect", "Good", "Fair", "Poor"]
✅ GOOD (contextual): ["Right first step", "Good sequence", "Wrong order", "Needs rethinking"]

Generate the evaluation questions with analysis now:"""

    return prompt

def build_survey_generator_prompt(questions: List[Dict], domain: str, burning_issues: List[Dict]) -> str:
    """Build the enhanced prompt for the Survey Generator agent"""
    question_types = {}
    for q in questions:
        q_type = q.get('type', 'unknown')
        question_types[q_type] = question_types.get(q_type, 0) + 1

    open_ended_count = question_types.get('open', 0)
    multiple_choice_count = question_types.get('multiple_choice', 0)
    scale_count = question_types.get('scale', 0)
    total_questions = len(questions)

    # Sample questions for context
    sample_questions = "\n".join([f"  • {q.get('text', '')[:80]}..." for q in questions[:3]])

    # Burning issues context
    issues_text = "\n".join([f"  • {issue.get('title', '')}" for issue in burning_issues[:3]])

    prompt = f"""You are an expert Meta-Feedback Generator specializing in survey design evaluation for startup validation research.

═══════════════════════════════════════════════════════════════════════════════
AGENT BEING EVALUATED: Survey Generator Agent
DOMAIN: {domain.upper()}
TARGET VALIDATION AREAS:
{issues_text}
═══════════════════════════════════════════════════════════════════════════════

CONTEXT & AGENT OUTPUT:
The Survey Generator created a survey to validate the burning issues. Here's the survey composition:

SURVEY STATISTICS:
• Total Questions: {total_questions}
• Open-ended: {open_ended_count} ({open_ended_count/total_questions*100:.0f}%)
• Multiple choice: {multiple_choice_count}
• Scale/rating: {scale_count}

SAMPLE QUESTIONS GENERATED:
{sample_questions}

═══════════════════════════════════════════════════════════════════════════════
YOUR TASK:
Generate 1-2 targeted evaluation questions that assess the quality and effectiveness of the survey design.

EVALUATION CRITERIA TO CONSIDER:
1. **Relevance**: Do questions directly test the burning issue hypotheses?
2. **Clarity**: Are questions easy to understand and unambiguous?
3. **Balance**: Is the mix of question types appropriate?
4. **Bias**: Are questions neutral or leading?
5. **Actionability**: Will answers provide actionable validation data?
6. **Length**: Is the survey appropriately scoped for respondent attention span?

WHY SURVEY QUALITY MATTERS:
Poor survey design leads to:
- Biased or misleading validation data
- Low response rates
- False confidence in wrong directions
- Wasted time with unusable feedback

EXAMPLES OF EXCELLENT EVALUATION QUESTIONS:
✓ "How relevant are these questions to your validation needs?" (core relevance)
✓ "Question clarity and neutrality?" (quality assessment)
✓ "Will this survey give you actionable validation data?" (outcome focus)

EXAMPLES OF POOR EVALUATION QUESTIONS:
✗ "What questions should we ask?" (doing the AI's job)
✗ "Do you like surveys?" (not evaluating output)
✗ "How long should a survey be in general for customer research?" (too general)

CHAIN-OF-THOUGHT REASONING:
Consider:
1. What's the biggest risk in bad survey design? (biased/irrelevant questions → false validation)
2. Can the founder judge question quality without being a survey expert?
3. Should we focus on relevance, clarity, or balance?

ANALYSIS & OUTPUT FORMAT:

STEP 1: Analyze survey design quality
Before generating questions, analyze:
- Do questions TEST hypotheses or just collect generic data?
- Is there leading/biased language?
- Is the question mix appropriate (open vs closed)?
- Are guardrail questions properly placed?

STEP 2: Generate output in this EXACT JSON format:
{{
  "questions": [
    {{
      "id": "q1",
      "text": "Specific, actionable evaluation question (max 150 chars)",
      "type": "multiple_choice",
      "focus_area": "hypothesis_testing_quality|question_clarity|research_design|bias_detection|respondent_experience",
      "options": [
        {{"value": "excellent_design", "label": "Context-specific label for EXCELLENT survey design"}},
        {{"value": "good_design", "label": "Context-specific label for GOOD design"}},
        {{"value": "fair_design", "label": "Context-specific label for FAIR design"}},
        {{"value": "poor_design", "label": "Context-specific label for POOR design"}}
      ],
      "rl_value": 0.87
    }},
    {{
      "id": "q2",
      "text": "Optional second question (max 150 chars)",
      "type": "multiple_choice",
      "focus_area": "hypothesis_testing_quality|question_clarity|research_design|bias_detection|respondent_experience",
      "options": [
        {{"value": "excellent", "label": "Context-specific label"}},
        {{"value": "good", "label": "Context-specific label"}},
        {{"value": "fair", "label": "Context-specific label"}},
        {{"value": "poor", "label": "Context-specific label"}}
      ],
      "rl_value": 0.81
    }}
  ],
  "analysis_summary": "2-3 sentence summary of survey quality. Example: 'Survey includes {total_questions} questions with good hypothesis-testing focus. However, Q3 may be leading. Open-ended ratio ({open_ended_count}/{total_questions}) is appropriate. Demographic guardrails properly placed. Overall: Good execution with minor bias risks.'"
}}

IMPORTANT CONSTRAINTS:
- MUST generate EXACTLY 2 questions (not 1, not more than 2)
- Each question MUST be under 150 characters
- Use "focus_area" with specific names like "hypothesis_testing_quality"
- Generate OPTIONS directly (context-specific to survey quality)
- Include "analysis_summary" explaining survey design quality
- RL values should reflect validation criticality (0.75-0.90 range)
- Return ONLY valid JSON

OPTION GENERATION RULES:
❌ BAD (generic): ["Very relevant", "Somewhat relevant", "Not relevant"]
✅ GOOD (contextual): ["Tests hypotheses", "Mostly tests", "Generic questions", "Doesn't test"]

Generate the evaluation questions with analysis now:"""

    return prompt

def build_360_report_prompt(verdict: str, composite_score: int, confidence: str) -> str:
    """Build the enhanced prompt for the 360 Report agent"""

    # Parse verdict to provide context
    verdict_interpretation = ""
    if "pivot" in verdict.lower():
        verdict_interpretation = "RECOMMENDATION: Pivot or major course correction needed"
    elif "proceed" in verdict.lower() or "go" in verdict.lower():
        verdict_interpretation = "RECOMMENDATION: Move forward with current direction"
    elif "pause" in verdict.lower() or "wait" in verdict.lower():
        verdict_interpretation = "RECOMMENDATION: Pause and gather more data"
    elif "kill" in verdict.lower() or "stop" in verdict.lower():
        verdict_interpretation = "RECOMMENDATION: Stop pursuing this idea"
    else:
        verdict_interpretation = f"RECOMMENDATION: {verdict}"

    prompt = f"""You are an expert Meta-Feedback Generator specializing in evaluating strategic business recommendations for startups.

═══════════════════════════════════════════════════════════════════════════════
AGENT BEING EVALUATED: 360 Report Agent (Final Verdict Generator)
ROLE: Synthesizes all validation data into a strategic go/no-go/pivot recommendation
═══════════════════════════════════════════════════════════════════════════════

CONTEXT & AGENT OUTPUT:
The 360 Report agent analyzed ALL validation data (market research, customer feedback, expert opinions, competitive analysis) and synthesized this final verdict:

{verdict_interpretation}
COMPOSITE VALIDATION SCORE: {composite_score}/100
CONFIDENCE LEVEL: {confidence}%

This is the HIGHEST-STAKES output in the entire pipeline - it directly influences whether the founder:
• Invests months/years pursuing this idea
• Pivots to a different approach
• Abandons the concept entirely

═══════════════════════════════════════════════════════════════════════════════
YOUR TASK:
Generate 1-2 critical evaluation questions that assess whether the AI's strategic recommendation is sound.

EVALUATION CRITERIA TO CONSIDER:
1. **Verdict Accuracy**: Does the recommendation align with the founder's ground truth?
2. **Evidence Quality**: Is the verdict well-supported by the validation data?
3. **Confidence Calibration**: Is the AI appropriately certain/uncertain given the data?
4. **Actionability**: Does the founder know what to do next based on this verdict?
5. **Risk Assessment**: Did the AI identify critical risks correctly?

WHY 360 REPORT QUALITY MATTERS (CRITICAL):
This verdict determines:
- Whether founders waste 6-12 months on a doomed idea
- Whether they abandon a viable opportunity too early
- Resource allocation decisions (time, money, team focus)
- Investor/stakeholder confidence

A bad verdict = potentially catastrophic consequences

EXAMPLES OF EXCELLENT EVALUATION QUESTIONS:
✓ "Does this verdict match your ground-truth understanding?" (accuracy)
✓ "How well does our recommendation align with your thinking?" (strategic alignment)
✓ "Confidence in our {verdict.lower()} recommendation?" (confidence assessment)

EXAMPLES OF POOR EVALUATION QUESTIONS:
✗ "What should you do with your startup?" (asking user to do AI's job)
✗ "Are you happy with the report?" (too vague, emotional)
✗ "How many validation lenses did we use effectively in our analysis?" (wrong focus)

CHAIN-OF-THOUGHT REASONING:
Think deeply:
1. The founder has real-world context we don't - can they judge if verdict is right?
2. Should we ask about verdict alignment or confidence level?
3. What failure mode are we most worried about? (false positive vs false negative)

ANALYSIS & OUTPUT FORMAT:

STEP 1: Analyze verdict quality
Before generating questions, analyze:
- Is the score (0-100) justified by the data shown?
- Does the decision (Proceed/Pivot/Pause/Kill) match the score?
- Is confidence appropriately calibrated?
- Are top risks accurately identified?

STEP 2: Generate output in this EXACT JSON format:
{{
  "questions": [
    {{
      "id": "q1",
      "text": "Critical evaluation question (max 150 chars)",
      "type": "multiple_choice",
      "focus_area": "verdict_accuracy|strategic_alignment|evidence_quality|confidence_calibration|risk_identification",
      "options": [
        {{"value": "fully_aligned", "label": "Context-specific label for FULL ALIGNMENT"}},
        {{"value": "mostly_aligned", "label": "Context-specific label for MOSTLY ALIGNED"}},
        {{"value": "partly_aligned", "label": "Context-specific label for PARTIAL ALIGNMENT"}},
        {{"value": "misaligned", "label": "Context-specific label for MISALIGNMENT"}}
      ],
      "rl_value": 0.95
    }},
    {{
      "id": "q2",
      "text": "Optional second question (max 150 chars)",
      "type": "multiple_choice",
      "focus_area": "verdict_accuracy|strategic_alignment|evidence_quality|confidence_calibration|risk_identification",
      "options": [
        {{"value": "excellent", "label": "Context-specific label"}},
        {{"value": "good", "label": "Context-specific label"}},
        {{"value": "fair", "label": "Context-specific label"}},
        {{"value": "poor", "label": "Context-specific label"}}
      ],
      "rl_value": 0.90
    }}
  ],
  "analysis_summary": "2-3 sentence summary of verdict quality. Example: 'Verdict shows '{verdict}' with {composite_score} score and {confidence}% confidence. Decision matches score appropriately. However, risk identification could be more specific. Confidence calibration is reasonable given data completeness.'"
}}

IMPORTANT CONSTRAINTS:
- MUST generate EXACTLY 2 questions (not 1, not more than 2)
- Each question MUST be under 150 characters
- Use "focus_area" with specific names like "verdict_accuracy"
- Generate OPTIONS directly (context-specific to verdict quality)
- Include "analysis_summary" explaining verdict quality assessment
- RL values should be HIGH (0.88-0.98 range) - most critical evaluation
- Return ONLY valid JSON

OPTION GENERATION RULES:
❌ BAD (generic): ["Yes, confirms", "Makes me reconsider", "Disagree"]
✅ GOOD (contextual): ["Matches my view", "Partly matches", "Conflicts", "Totally wrong"]

Generate the evaluation questions with analysis now:"""

    return prompt

def build_social_lens_prompt(buzz_score: int, buzz_level: str, sources: List[str]) -> str:
    """Build the enhanced prompt for the Social Lens agent"""

    # Categorize buzz level for context
    buzz_interpretation = ""
    if buzz_score >= 80:
        buzz_interpretation = "HIGH buzz - significant social/market conversation"
    elif buzz_score >= 50:
        buzz_interpretation = "MODERATE buzz - growing interest"
    elif buzz_score >= 25:
        buzz_interpretation = "LOW buzz - emerging conversation"
    else:
        buzz_interpretation = "MINIMAL buzz - very limited discussion"

    # Sample sources for context
    source_sample = "\n".join([f"  • {source}" for source in sources[:5]])

    prompt = f"""You are an expert Meta-Feedback Generator specializing in social media and market sentiment analysis evaluation.

═══════════════════════════════════════════════════════════════════════════════
AGENT BEING EVALUATED: Social Synth Agent (Social Lens)
ROLE: Analyzes social media, forums, news, and online discourse to gauge market buzz
═══════════════════════════════════════════════════════════════════════════════

CONTEXT & AGENT OUTPUT:
The Social Synth agent analyzed online conversations and produced this buzz assessment:

BUZZ ANALYSIS:
• Overall Score: {buzz_score}/100
• Classification: {buzz_level}
• Interpretation: {buzz_interpretation}
• Sources Analyzed: {len(sources)}

SAMPLE DATA SOURCES:
{source_sample}

═══════════════════════════════════════════════════════════════════════════════
YOUR TASK:
Generate 1-2 focused evaluation questions that assess the accuracy of the social buzz analysis.

EVALUATION CRITERIA TO CONSIDER:
1. **Score Accuracy**: Does the buzz score match the founder's perception of market conversation?
2. **Source Quality**: Are the analyzed sources relevant and authoritative?
3. **Sentiment Calibration**: Is the AI over/under-estimating the buzz level?
4. **Context Understanding**: Did the AI correctly interpret social signals vs noise?
5. **Timeliness**: Does the analysis capture current buzz or outdated trends?

WHY SOCIAL LENS ACCURACY MATTERS:
Social buzz misinterpretation leads to:
- False validation (thinking there's demand when there isn't)
- Missed opportunities (underestimating real market interest)
- Poor timing decisions (launching too early/late)
- Misallocated marketing resources

EXAMPLES OF EXCELLENT EVALUATION QUESTIONS:
✓ "Does our buzz score match your market perception?" (accuracy check)
✓ "Quality of our social signal sources?" (source validation)
✓ "Did we capture the real conversation level?" (sentiment calibration)

EXAMPLES OF POOR EVALUATION QUESTIONS:
✗ "How much buzz should there be?" (asking founder to do analysis)
✗ "Do you use social media?" (not evaluating output)
✗ "What is the best social media platform for startups in general?" (too general)

CHAIN-OF-THOUGHT REASONING:
Consider:
1. Can the founder validate buzz level from their own market experience?
2. Is it more critical to evaluate score accuracy or source quality?
3. Should we ask about over/under-estimation of buzz?

ANALYSIS & OUTPUT FORMAT:

STEP 1: Analyze buzz assessment quality
Before generating questions, analyze:
- Is the buzz score (0-100) realistic for this space?
- Are sources authoritative and relevant?
- Do talking points capture real signals vs noise?
- Is sentiment analysis nuanced?

STEP 2: Generate output in this EXACT JSON format:
{{
  "questions": [
    {{
      "id": "q1",
      "text": "Focused evaluation question (max 150 chars)",
      "type": "multiple_choice",
      "focus_area": "buzz_score_accuracy|source_quality|signal_detection|sentiment_calibration|trend_identification",
      "options": [
        {{"value": "accurate", "label": "Context-specific label for ACCURATE assessment"}},
        {{"value": "slightly_high", "label": "Context-specific label for SLIGHTLY HIGH"}},
        {{"value": "slightly_low", "label": "Context-specific label for SLIGHTLY LOW"}},
        {{"value": "way_off", "label": "Context-specific label for WAY OFF"}}
      ],
      "rl_value": 0.86
    }},
    {{
      "id": "q2",
      "text": "Optional second question (max 150 chars)",
      "type": "multiple_choice",
      "focus_area": "buzz_score_accuracy|source_quality|signal_detection|sentiment_calibration|trend_identification",
      "options": [
        {{"value": "excellent", "label": "Context-specific label"}},
        {{"value": "good", "label": "Context-specific label"}},
        {{"value": "fair", "label": "Context-specific label"}},
        {{"value": "poor", "label": "Context-specific label"}}
      ],
      "rl_value": 0.82
    }}
  ],
  "analysis_summary": "2-3 sentence summary of buzz analysis quality. Example: 'Buzz score of {buzz_score}/100 classified as '{buzz_level}' seems appropriate given {len(sources)} sources. However, heavy reliance on academic sources may miss real-world signals. Source quality: High. Signal-to-noise ratio: Good.'"
}}

IMPORTANT CONSTRAINTS:
- MUST generate EXACTLY 2 questions (not 1, not more than 2)
- Each question MUST be under 150 characters
- Use "focus_area" with specific names like "buzz_score_accuracy"
- Generate OPTIONS directly (context-specific to buzz assessment)
- Include "analysis_summary" explaining buzz analysis quality
- RL values should reflect buzz analysis importance (0.78-0.90 range)
- Return ONLY valid JSON

OPTION GENERATION RULES:
❌ BAD (generic): ["Spot on", "Too high", "Too low", "Not sure"]
✅ GOOD (contextual): ["Matches reality", "Bit inflated", "Bit conservative", "Way off"]

Generate the evaluation questions with analysis now:"""

    return prompt

def build_match_maker_prompt(selected_match: Dict, match_score: float) -> str:
    """Build the enhanced prompt for the Match-Maker agent"""
    match_name = selected_match.get('name', 'Unknown')
    match_type = selected_match.get('type', 'Unknown')
    match_expertise = selected_match.get('expertise', [])
    match_background = selected_match.get('background', '')[:150]

    # Build expertise context
    expertise_text = ", ".join(match_expertise[:3]) if match_expertise else "Not specified"

    prompt = f"""You are an expert Meta-Feedback Generator specializing in evaluating AI-powered matching systems for startups.

═══════════════════════════════════════════════════════════════════════════════
AGENT BEING EVALUATED: Match-Maker Agent
ROLE: Matches founders with relevant SMEs, co-founders, or survey respondents
═══════════════════════════════════════════════════════════════════════════════

CONTEXT & AGENT OUTPUT:
The Match-Maker agent analyzed the founder's needs and recommended this match:

RECOMMENDED MATCH:
• Name/Profile: {match_name}
• Match Type: {match_type}
• Match Score: {match_score:.0%}
• Expertise Areas: {expertise_text}
• Background: {match_background}...

This match is intended to help the founder with validation, expertise, or execution.

═══════════════════════════════════════════════════════════════════════════════
YOUR TASK:
Generate 1-2 targeted evaluation questions that assess the quality and relevance of this match.

EVALUATION CRITERIA TO CONSIDER:
1. **Relevance**: Does this person/profile align with the founder's actual needs?
2. **Expertise Fit**: Do they have the right domain knowledge/skills?
3. **Match Quality**: Is this a high-value connection vs generic suggestion?
4. **Understanding**: Did the AI correctly interpret what kind of help the founder needs?
5. **Actionability**: Can the founder meaningfully engage with this match?

WHY MATCH QUALITY MATTERS:
Poor matching leads to:
- Wasted time on irrelevant conversations
- Missed opportunities with better-fit experts
- Founder frustration and disengagement
- Loss of trust in the AI system

Good matching enables:
- High-value validation conversations
- Strategic advisory relationships
- Efficient use of founder's limited time

EXAMPLES OF EXCELLENT EVALUATION QUESTIONS:
✓ "How relevant is this match to your needs?" (direct relevance)
✓ "Match quality for your validation goals?" (quality assessment)
✓ "Does this {match_type} have the right expertise?" (expertise fit)

EXAMPLES OF POOR EVALUATION QUESTIONS:
✗ "Who should we match you with?" (asking founder to do AI's job)
✗ "Do you like meeting new people?" (not evaluating match quality)
✗ "How do you typically network with industry experts?" (wrong focus)

CHAIN-OF-THOUGHT REASONING:
Consider:
1. Can the founder judge match relevance from the profile info?
2. Is expertise fit or overall relevance more critical to evaluate?
3. Should we ask about this specific match or the AI's understanding?

ANALYSIS & OUTPUT FORMAT:

STEP 1: Analyze match quality
Before generating questions, analyze:
- Is expertise directly relevant to burning issues?
- Is match score (0.0-1.0) well-justified?
- Does match type align with founder needs?
- Is this a high-value connection or generic suggestion?

STEP 2: Generate output in this EXACT JSON format:
{{
  "questions": [
    {{
      "id": "q1",
      "text": "Targeted evaluation question (max 150 chars)",
      "type": "multiple_choice",
      "focus_area": "match_relevance|expertise_fit|match_precision|value_assessment|profile_quality",
      "options": [
        {{"value": "perfect_match", "label": "Context-specific label for PERFECT MATCH"}},
        {{"value": "good_match", "label": "Context-specific label for GOOD MATCH"}},
        {{"value": "fair_match", "label": "Context-specific label for FAIR MATCH"}},
        {{"value": "poor_match", "label": "Context-specific label for POOR MATCH"}}
      ],
      "rl_value": 0.89
    }},
    {{
      "id": "q2",
      "text": "Optional second question (max 150 chars)",
      "type": "multiple_choice",
      "focus_area": "match_relevance|expertise_fit|match_precision|value_assessment|profile_quality",
      "options": [
        {{"value": "excellent", "label": "Context-specific label"}},
        {{"value": "good", "label": "Context-specific label"}},
        {{"value": "fair", "label": "Context-specific label"}},
        {{"value": "poor", "label": "Context-specific label"}}
      ],
      "rl_value": 0.84
    }}
  ],
  "analysis_summary": "2-3 sentence summary of match quality. Example: 'Matched {match_type} has relevant expertise in {expertise_text} (aligns with burning issues). Match score of {match_score:.0%} seems justified. However, profile completeness could be better. Match relevance: High. Value potential: Strong.'"
}}

IMPORTANT CONSTRAINTS:
- MUST generate EXACTLY 2 questions (not 1, not more than 2)
- Each question MUST be under 150 characters
- Use "focus_area" with specific names like "match_relevance"
- Generate OPTIONS directly (context-specific to match quality)
- Include "analysis_summary" explaining match quality assessment
- RL values should reflect matching importance (0.80-0.92 range)
- Return ONLY valid JSON

OPTION GENERATION RULES:
❌ BAD (generic): ["Perfect", "Good", "Okay", "Poor"]
✅ GOOD (contextual): ["Direct fit", "Good overlap", "Partial fit", "Weak fit"]

Generate the evaluation questions with analysis now:"""

    return prompt

def build_survey_analysis_prompt(batch_insights: Dict, clusters: List[Dict],
                                 sentiment_analysis: Dict, executive_summary: Dict) -> str:
    """Build the enhanced prompt for the Survey Analysis agent"""

    # Extract key metrics
    key_themes = batch_insights.get('key_themes', [])
    pain_points = batch_insights.get('pain_points', [])
    opportunities = batch_insights.get('opportunities', [])

    num_clusters = len(clusters)
    primary_segment = executive_summary.get('primary_segment', 'Unknown')
    market_opportunity = executive_summary.get('market_opportunity', 'Unknown')

    # Build themes text
    themes_text = "\n".join([f"  • {theme}" for theme in key_themes[:3]]) if key_themes else "  • No themes identified"

    # Build clusters summary
    clusters_text = ""
    for i, cluster in enumerate(clusters[:3], 1):
        title = cluster.get('title', f'Segment {i}')
        pain_intensity = cluster.get('pain_intensity', 0)
        clusters_text += f"  {i}. {title} (pain intensity: {pain_intensity}/10)\n"

    # Sentiment breakdown
    overall_sentiment = sentiment_analysis.get('overall_sentiment', {})

    prompt = f"""You are an expert Meta-Feedback Generator specializing in evaluating AI-powered survey analysis for startups.

═══════════════════════════════════════════════════════════════════════════════
AGENT BEING EVALUATED: Survey Analysis Agent
ROLE: Analyzes batch survey responses to extract customer insights, segments, and opportunities
═══════════════════════════════════════════════════════════════════════════════

CONTEXT & AGENT OUTPUT:
The Survey Analysis agent processed survey responses and generated this analysis:

BATCH INSIGHTS:
Key Themes Identified:
{themes_text}

Top Pain Points: {len(pain_points)} identified
Opportunities: {len(opportunities)} identified

CUSTOMER SEGMENTS:
{clusters_text}
Total Segments: {num_clusters}

SENTIMENT ANALYSIS:
Overall Sentiment: {overall_sentiment}

EXECUTIVE SUMMARY:
• Primary Target Segment: {primary_segment}
• Market Opportunity: {market_opportunity}

═══════════════════════════════════════════════════════════════════════════════
YOUR TASK:
Generate 1-2 critical evaluation questions that assess the quality and actionability of this survey analysis.

EVALUATION CRITERIA TO CONSIDER:
1. **Insight Depth**: Are the identified themes/patterns truly insightful vs superficial?
2. **Segmentation Quality**: Are customer segments meaningfully distinct and actionable?
3. **Pain Point Accuracy**: Do pain points reflect real customer frustrations?
4. **Opportunity Identification**: Are opportunities realistic and specific?
5. **Sentiment Calibration**: Is sentiment analysis accurate and nuanced?
6. **Executive Summary**: Is the strategic guidance clear and actionable?

WHY SURVEY ANALYSIS QUALITY MATTERS:
Poor survey analysis leads to:
- Misidentified customer segments (wasting marketing budget)
- False confidence in wrong pain points
- Missed real opportunities hidden in responses
- Strategic decisions based on surface-level patterns

EXAMPLES OF EXCELLENT EVALUATION QUESTIONS:
✓ "Do these customer segments match your actual market?" (segmentation accuracy)
✓ "How well did we identify your customers' real pain points?" (insight quality)
✓ "Actionability of our opportunity recommendations?" (strategic value)

EXAMPLES OF POOR EVALUATION QUESTIONS:
✗ "What are your customer segments?" (asking user to do AI's job)
✗ "Do you like survey analysis?" (not evaluating output quality)

ANALYSIS & OUTPUT FORMAT:

STEP 1: Analyze survey analysis quality
- Are segments based on meaningful behavioral/attitudinal differences?
- Are themes actually patterns vs just word frequency?
- Do pain points have clear intensity/priority ordering?
- Are opportunities specific and testable?

STEP 2: Generate output in this EXACT JSON format:
{{
  "questions": [
    {{
      "id": "q1",
      "text": "Critical evaluation question (max 150 chars)",
      "type": "multiple_choice",
      "focus_area": "segment_accuracy|insight_depth|pain_point_identification|opportunity_quality|sentiment_calibration|strategic_clarity",
      "options": [
        {{"value": "matches_reality", "label": "Context-specific label for HIGH ACCURACY"}},
        {{"value": "mostly_accurate", "label": "Context-specific label for MOSTLY ACCURATE"}},
        {{"value": "some_gaps", "label": "Context-specific label for PARTIAL ACCURACY"}},
        {{"value": "way_off", "label": "Context-specific label for INACCURATE"}}
      ],
      "rl_value": 0.91
    }},
    {{
      "id": "q2",
      "text": "Optional second question (max 150 chars)",
      "type": "multiple_choice",
      "focus_area": "segment_accuracy|insight_depth|pain_point_identification|opportunity_quality|sentiment_calibration|strategic_clarity",
      "options": [
        {{"value": "actionable_opps", "label": "Context-specific label"}},
        {{"value": "mostly_actionable", "label": "Context-specific label"}},
        {{"value": "vague_opps", "label": "Context-specific label"}},
        {{"value": "not_actionable", "label": "Context-specific label"}}
      ],
      "rl_value": 0.87
    }}
  ],
  "analysis_summary": "2-3 sentence summary of survey analysis quality. Example: 'Identified {num_clusters} customer segments with clear differentiation. Primary segment ({primary_segment}) aligns with market opportunity. Pain points show good specificity. Overall: Strong insights.'"
}}

IMPORTANT CONSTRAINTS:
- MUST generate EXACTLY 2 questions
- Each question MUST be under 150 characters
- Use "focus_area" with specific names
- Generate OPTIONS directly (context-specific)
- Include "analysis_summary"
- RL values: 0.85-0.93 range
- Return ONLY valid JSON

Generate the evaluation questions with analysis now:"""

    return prompt

def build_meeting_summary_prompt(bullet_points: str, persona_type: str, meeting_id: str) -> str:
    """Build the enhanced prompt for the Meeting Summary agent"""

    # Extract a brief preview of bullet points
    bullet_preview = bullet_points[:200] + "..." if len(bullet_points) > 200 else bullet_points

    prompt = f"""You are an expert Meta-Feedback Generator specializing in evaluating AI-generated meeting preparation summaries.

═══════════════════════════════════════════════════════════════════════════════
AGENT BEING EVALUATED: Meeting Summary Agent
ROLE: Generates concise, actionable meeting preparation summaries for {persona_type} conversations
MEETING ID: {meeting_id}
═══════════════════════════════════════════════════════════════════════════════

CONTEXT & AGENT OUTPUT:
The Meeting Summary agent analyzed the meeting context and generated this summary:

{persona_type} MEETING SUMMARY PREVIEW:
{bullet_preview}

This summary is designed to help founders prepare efficiently for high-value conversations with {persona_type}s.

═══════════════════════════════════════════════════════════════════════════════
YOUR TASK:
Generate 1-2 focused evaluation questions that assess the usefulness and accuracy of this meeting summary.

EVALUATION CRITERIA TO CONSIDER:
1. **Relevance**: Are the bullet points relevant to the actual meeting context?
2. **Actionability**: Can the founder act on these preparation points?
3. **Completeness**: Are critical preparation areas missing?
4. **Clarity**: Are points clear and specific vs vague?
5. **Time Efficiency**: Does this save time vs overwhelm with detail?

WHY MEETING SUMMARY QUALITY MATTERS:
Poor summaries lead to unprepared founders wasting expert time.
Good summaries enable focused, productive conversations.

EXAMPLES OF EXCELLENT EVALUATION QUESTIONS:
✓ "How relevant are these prep points to your actual meeting?" (relevance check)
✓ "Completeness of our {persona_type} meeting summary?" (coverage assessment)
✓ "Can you act on these preparation guidelines?" (actionability test)

EXAMPLES OF POOR EVALUATION QUESTIONS:
✗ "What should you prepare for the meeting?" (asking user to do AI's job)

ANALYSIS & OUTPUT FORMAT:

STEP 2: Generate output in this EXACT JSON format:
{{
  "questions": [
    {{
      "id": "q1",
      "text": "Focused evaluation question (max 150 chars)",
      "type": "multiple_choice",
      "focus_area": "prep_relevance|summary_completeness|actionability|clarity|prioritization|time_efficiency",
      "options": [
        {{"value": "right_on_target", "label": "Context-specific label for HIGH RELEVANCE"}},
        {{"value": "mostly_helpful", "label": "Context-specific label for MOSTLY RELEVANT"}},
        {{"value": "some_value", "label": "Context-specific label for SOMEWHAT RELEVANT"}},
        {{"value": "off_target", "label": "Context-specific label for NOT RELEVANT"}}
      ],
      "rl_value": 0.89
    }},
    {{
      "id": "q2",
      "text": "Optional second question (max 150 chars)",
      "type": "multiple_choice",
      "focus_area": "prep_relevance|summary_completeness|actionability|clarity|prioritization|time_efficiency",
      "options": [
        {{"value": "comprehensive", "label": "Context-specific label"}},
        {{"value": "good_coverage", "label": "Context-specific label"}},
        {{"value": "missing_some", "label": "Context-specific label"}},
        {{"value": "incomplete", "label": "Context-specific label"}}
      ],
      "rl_value": 0.84
    }}
  ],
  "analysis_summary": "2-3 sentence summary of meeting summary quality. Example: 'Generated {persona_type} meeting prep with specific points. Bullet points are actionable and time-efficient. Overall: Good execution.'"
}}

IMPORTANT CONSTRAINTS:
- MUST generate EXACTLY 2 questions
- Each question MUST be under 150 characters
- Use "focus_area" with specific names
- Generate OPTIONS directly
- Include "analysis_summary"
- RL values: 0.82-0.91 range
- Return ONLY valid JSON

Generate the evaluation questions with analysis now:"""

    return prompt

def build_transcript_summary_prompt(insights: List[Dict], quotes: List[Dict], persona_type: str) -> str:
    """Build the enhanced prompt for the Transcript Summary agent"""

    # Calculate insight metrics
    num_insights = len(insights)
    high_confidence_insights = len([i for i in insights if i.get('confidence_score', 0) >= 8.0])
    high_impact_insights = len([i for i in insights if i.get('impact_level', '').lower() in ['high_impact', 'game_changer']])

    # Extract sample insights
    sample_insights = "\n".join([
        f"  • {insight.get('insight', '')[:80]}... (confidence: {insight.get('confidence_score', 0):.1f}, impact: {insight.get('impact_level', 'unknown')})"
        for insight in insights[:3]
    ])

    # Quote metrics
    num_quotes = len(quotes)
    high_relevance_quotes = len([q for q in quotes if q.get('relevance_score', 0) >= 8.0])

    prompt = f"""You are an expert Meta-Feedback Generator specializing in evaluating AI-powered transcript analysis for startup validation conversations.

═══════════════════════════════════════════════════════════════════════════════
AGENT BEING EVALUATED: Transcript Summary Agent
ROLE: Analyzes meeting transcripts to extract key insights and supporting quotes
PERSONA TYPE: {persona_type.upper()}
═══════════════════════════════════════════════════════════════════════════════

CONTEXT & AGENT OUTPUT:
The Transcript Summary agent processed the meeting transcript and generated:

INSIGHTS EXTRACTED:
Total Insights: {num_insights}
High Confidence (≥8.0): {high_confidence_insights}
High Impact: {high_impact_insights}

Sample Insights:
{sample_insights}

SUPPORTING QUOTES:
Total Quotes: {num_quotes}
High Relevance (≥8.0): {high_relevance_quotes}

═══════════════════════════════════════════════════════════════════════════════
YOUR TASK:
Generate 1-2 sharp evaluation questions that assess the quality and usefulness of this transcript analysis.

EVALUATION CRITERIA TO CONSIDER:
1. **Insight Quality**: Are insights genuinely valuable vs obvious/superficial?
2. **Confidence Calibration**: Are confidence scores appropriately set?
3. **Impact Assessment**: Are high-impact insights correctly identified?
4. **Quote Relevance**: Do quotes effectively support insights?
5. **Completeness**: Are key moments from the conversation captured?

WHY TRANSCRIPT ANALYSIS QUALITY MATTERS:
Poor analysis leads to missing critical validation signals from expert conversations.
Good analysis enables efficient knowledge extraction from expert time.

EXAMPLES OF EXCELLENT EVALUATION QUESTIONS:
✓ "How accurately did we capture the key insights?" (insight quality)
✓ "Are these insights truly valuable vs obvious?" (depth assessment)

EXAMPLES OF POOR EVALUATION QUESTIONS:
✗ "What were the key insights from your meeting?" (asking user to do AI's job)

ANALYSIS & OUTPUT FORMAT:

STEP 2: Generate output in this EXACT JSON format:
{{
  "questions": [
    {{
      "id": "q1",
      "text": "Sharp evaluation question (max 150 chars)",
      "type": "multiple_choice",
      "focus_area": "insight_accuracy|insight_depth|confidence_calibration|impact_assessment|quote_relevance|completeness|persona_detection",
      "options": [
        {{"value": "spot_on", "label": "Context-specific label for EXCELLENT ACCURACY"}},
        {{"value": "mostly_right", "label": "Context-specific label for GOOD ACCURACY"}},
        {{"value": "partially_right", "label": "Context-specific label for PARTIAL ACCURACY"}},
        {{"value": "missed_key_points", "label": "Context-specific label for POOR ACCURACY"}}
      ],
      "rl_value": 0.92
    }},
    {{
      "id": "q2",
      "text": "Optional second question (max 150 chars)",
      "type": "multiple_choice",
      "focus_area": "insight_accuracy|insight_depth|confidence_calibration|impact_assessment|quote_relevance|completeness|persona_detection",
      "options": [
        {{"value": "well_calibrated", "label": "Context-specific label"}},
        {{"value": "mostly_calibrated", "label": "Context-specific label"}},
        {{"value": "overconfident", "label": "Context-specific label"}},
        {{"value": "underconfident", "label": "Context-specific label"}}
      ],
      "rl_value": 0.88
    }}
  ],
  "analysis_summary": "2-3 sentence summary of transcript analysis quality. Example: 'Extracted {num_insights} insights from {persona_type} conversation with {high_confidence_insights} high-confidence findings. Impact assessment shows {high_impact_insights} actionable insights. Overall: Strong analysis.'"
}}

IMPORTANT CONSTRAINTS:
- MUST generate EXACTLY 2 questions
- Each question MUST be under 150 characters
- Use "focus_area" with specific names
- Generate OPTIONS directly
- Include "analysis_summary"
- RL values: 0.86-0.94 range
- Return ONLY valid JSON

Generate the evaluation questions with analysis now:"""

    return prompt

def get_idea_capture_data(ai_request_id: str) -> Dict:
    """Get idea capture data from DynamoDB with improved error handling"""
    try:
        dynamodb, _, _ = init_aws_clients()
        if not dynamodb:
            st.error("❌ Cannot access DynamoDB. AWS clients not initialized properly.")
            return None
        
        table = dynamodb.Table(IDEAS_TABLE)
        logger.info(f"Attempting to fetch idea capture data for request ID: {ai_request_id}")
        
        response = table.get_item(Key={'ai_request_id': ai_request_id})
        
        if 'Item' in response:
            item = response['Item']
            logger.info(f"Found item for request ID: {ai_request_id}")
            
            # Parse the idea_response JSON
            idea_response_str = item.get('idea_response', '{}')
            if isinstance(idea_response_str, str):
                idea_response = json.loads(idea_response_str)
            else:
                idea_response = idea_response_str
            
            burning_issues = idea_response.get('finalize', {}).get('burningIssues', [])
            input_metadata = idea_response.get('input_metadata', {})
            domain = extract_domain(input_metadata)
            
            logger.info(f"Extracted {len(burning_issues)} burning issues for domain: {domain}")
            
            return {
                'burning_issues': burning_issues,
                'domain': domain,
                'input_metadata': input_metadata
            }
        else:
            logger.warning(f"No item found for request ID: {ai_request_id}")
            st.error(f"❌ No data found for request ID: {ai_request_id}")
            st.info("💡 Please verify that the request ID exists in the database.")
            return None
    except Exception as e:
        error_msg = f"Error fetching idea capture data: {str(e)}"
        logger.error(error_msg)
        st.error(f"❌ {error_msg}")
        
        # Provide more specific guidance based on error type
        if "ResourceNotFoundException" in str(e):
            st.error("🔍 The DynamoDB table might not exist or your credentials don't have access to it.")
            st.info("💡 Please check that the table name is correct and your IAM permissions include 'dynamodb:GetItem'.")
        elif "AccessDenied" in str(e):
            st.error("🔒 Access denied. Your AWS credentials may not have the necessary permissions.")
            st.info("💡 Please ensure your IAM role/user has permissions for DynamoDB operations.")
        
        return None

def get_lens_selector_data(ai_request_id: str) -> Dict:
    """Get lens selector data from DynamoDB with improved error handling"""
    try:
        dynamodb, _, _ = init_aws_clients()
        if not dynamodb:
            st.error("❌ Cannot access DynamoDB. AWS clients not initialized properly.")
            return None
        
        table = dynamodb.Table(IDEAS_TABLE)
        logger.info(f"Attempting to fetch lens selector data for request ID: {ai_request_id}")
        
        response = table.get_item(Key={'ai_request_id': ai_request_id})
        
        if 'Item' in response:
            item = response['Item']
            logger.info(f"Found item for request ID: {ai_request_id}")
            
            # Parse the lens_selector JSON
            lens_selector_str = item.get('lens_selector', '{}')
            if isinstance(lens_selector_str, str):
                lens_selector = json.loads(lens_selector_str)
            else:
                lens_selector = lens_selector_str
            
            results = lens_selector.get('result', {}).get('results', [])
            input_data = lens_selector.get('result', {}).get('input', {})
            domain = extract_domain(input_data)
            
            # Get burning issues summary
            idea_response_str = item.get('idea_response', '{}')
            if isinstance(idea_response_str, str):
                idea_response = json.loads(idea_response_str)
            else:
                idea_response = idea_response_str
                
            burning_issues = idea_response.get('finalize', {}).get('burningIssues', [])
            burning_issues_summary = [issue.get('title', '') for issue in burning_issues]
            
            logger.info(f"Extracted {len(results)} lens results for domain: {domain}")
            
            return {
                'results': results,
                'domain': domain,
                'burning_issues_summary': burning_issues_summary
            }
        else:
            logger.warning(f"No item found for request ID: {ai_request_id}")
            st.error(f"❌ No lens selector data found for request ID: {ai_request_id}")
            return None
    except Exception as e:
        error_msg = f"Error fetching lens selector data: {str(e)}"
        logger.error(error_msg)
        st.error(f"❌ {error_msg}")
        return None

def get_survey_generator_data(ai_request_id: str) -> Dict:
    """Get survey generator data from DynamoDB with improved error handling"""
    try:
        dynamodb, _, _ = init_aws_clients()
        if not dynamodb:
            st.error("❌ Cannot access DynamoDB. AWS clients not initialized properly.")
            return None
        
        table = dynamodb.Table(IDEAS_TABLE)
        logger.info(f"Attempting to fetch survey generator data for request ID: {ai_request_id}")
        
        response = table.get_item(Key={'ai_request_id': ai_request_id})
        
        if 'Item' in response:
            item = response['Item']
            logger.info(f"Found item for request ID: {ai_request_id}")
            
            # Parse the survey_generator JSON
            survey_generator_str = item.get('survey_generator', '{}')
            if isinstance(survey_generator_str, str):
                survey_generator = json.loads(survey_generator_str)
            else:
                survey_generator = survey_generator_str
                
            questions = survey_generator.get('questions', [])
            input_data = survey_generator.get('input', {})
            domain = extract_domain(input_data)
            burning_issues = input_data.get('burningIssues', [])
            
            logger.info(f"Extracted {len(questions)} survey questions for domain: {domain}")
            
            return {
                'questions': questions,
                'domain': domain,
                'burning_issues': burning_issues
            }
        else:
            logger.warning(f"No item found for request ID: {ai_request_id}")
            st.error(f"❌ No survey generator data found for request ID: {ai_request_id}")
            return None
    except Exception as e:
        error_msg = f"Error fetching survey generator data: {str(e)}"
        logger.error(error_msg)
        st.error(f"❌ {error_msg}")
        return None

def get_360_report_data(ai_request_id: str) -> Dict:
    """Get 360 report data from DynamoDB with improved error handling"""
    try:
        dynamodb, _, _ = init_aws_clients()
        if not dynamodb:
            st.error("❌ Cannot access DynamoDB. AWS clients not initialized properly.")
            return None

        table = dynamodb.Table(FINALIZE_REPORT_TABLE)
        logger.info(f"Attempting to fetch 360 report data for request ID: {ai_request_id}")
        
        response = table.get_item(Key={'ai_request_id': ai_request_id})

        if 'Item' in response:
            item = response['Item']
            logger.info(f"Found item for request ID: {ai_request_id}")

            # Handle both string and dict types for report_json
            report_json_raw = item.get('report_json', '{}')
            if isinstance(report_json_raw, str):
                report_json = json.loads(report_json_raw)
            else:
                report_json = report_json_raw

            verdict = report_json.get('verdict', {}).get('decision', 'Unknown')
            composite_score = report_json.get('verdict', {}).get('compositeScore', 0)
            confidence = report_json.get('verdict', {}).get('confidence', '0')

            logger.info(f"Extracted 360 report verdict: {verdict}, score: {composite_score}")

            return {
                'verdict': verdict,
                'composite_score': composite_score,
                'confidence': confidence
            }
        else:
            logger.warning(f"No item found for request ID: {ai_request_id}")
            st.error(f"❌ No 360 report data found for request ID: {ai_request_id}")
            return None
    except Exception as e:
        error_msg = f"Error fetching 360 report data: {str(e)}"
        logger.error(error_msg)
        st.error(f"❌ {error_msg}")
        
        # Provide specific guidance for common errors
        if "ResourceNotFoundException" in str(e):
            st.error("🔍 The FINALIZE_REPORT_TABLE might not exist or your credentials don't have access to it.")
            st.info("💡 Please check that the table name is correct and your IAM permissions include 'dynamodb:GetItem'.")
        
        return None

def get_social_lens_data(ai_request_id: str) -> Dict:
    """Get social lens data from DynamoDB with improved error handling"""
    try:
        dynamodb, _, _ = init_aws_clients()
        if not dynamodb:
            st.error("❌ Cannot access DynamoDB. AWS clients not initialized properly.")
            return None

        table = dynamodb.Table(SOCIAL_LENS_TABLE)
        logger.info(f"Attempting to fetch social lens data for request ID: {ai_request_id}")
        
        response = table.get_item(Key={'request_id': ai_request_id})

        if 'Item' in response:
            item = response['Item']
            logger.info(f"Found item for request ID: {ai_request_id}")

            # Handle both string and dict types for social_lens_analysis_json
            social_analysis_raw = item.get('social_lens_analysis_json', '{}')
            if isinstance(social_analysis_raw, str):
                social_analysis = json.loads(social_analysis_raw)
            else:
                social_analysis = social_analysis_raw

            buzz_score = social_analysis.get('overall_buzz_score', 0)
            buzz_level = social_analysis.get('buzz_level', 'Unknown')
            sources = social_analysis.get('sources', [])

            logger.info(f"Extracted social lens data: buzz score {buzz_score}, level {buzz_level}")

            return {
                'buzz_score': buzz_score,
                'buzz_level': buzz_level,
                'sources': sources
            }
        else:
            logger.warning(f"No item found for request ID: {ai_request_id}")
            st.error(f"❌ No social lens data found for request ID: {ai_request_id}")
            return None
    except Exception as e:
        error_msg = f"Error fetching social lens data: {str(e)}"
        logger.error(error_msg)
        st.error(f"❌ {error_msg}")
        
        # Provide specific guidance for common errors
        if "ResourceNotFoundException" in str(e):
            st.error("🔍 The SOCIAL_LENS_TABLE might not exist or your credentials don't have access to it.")
            st.info("💡 Please check that the table name is correct and your IAM permissions include 'dynamodb:GetItem'.")
        
        return None

def get_match_maker_data(ai_request_id: str, persona_type: str) -> Dict:
    """Get match maker data from S3 with improved error handling"""
    try:
        _, s3, _ = init_aws_clients()
        if not s3:
            st.error("❌ Cannot access S3. AWS clients not initialized properly.")
            return None
        
        # Determine S3 path
        persona_type = persona_type.lower()
        if persona_type == 'sme':
            prefix = 'match-making/SME-matching/'
            file_pattern = f'{ai_request_id}_sme_matching_'
        elif persona_type == 'founder':
            prefix = 'match-making/FOUNDER-matching/'
            file_pattern = f'{ai_request_id}_founder_matching_'
        elif persona_type == 'respondent':
            prefix = 'match-making/RESPONDENT-matching/'
            file_pattern = f'{ai_request_id}_respondent_matching_'
        else:
            st.error(f"❌ Invalid persona type: {persona_type}")
            return None
        
        logger.info(f"Attempting to fetch match maker data for request ID: {ai_request_id}, persona: {persona_type}")
        
        # List objects
        objects = s3.list_objects_v2(
            Bucket=MATCH_MAKING_BUCKET,
            Prefix=prefix
        )
        
        if 'Contents' not in objects:
            logger.warning(f"No objects found in S3 bucket {MATCH_MAKING_BUCKET} with prefix {prefix}")
            st.error(f"❌ No match maker data found for request ID: {ai_request_id}")
            st.info(f"💡 Checked S3 bucket '{MATCH_MAKING_BUCKET}' with prefix '{prefix}'")
            return None
        
        # Find the file with the highest number
        max_num = 0
        latest_file = None
        
        for obj in objects['Contents']:
            key = obj['Key']
            if file_pattern in key:
                match = re.search(r'_(\d+)\.json$', key)
                if match:
                    num = int(match.group(1))
                    if num > max_num:
                        max_num = num
                        latest_file = key
        
        if not latest_file:
            logger.warning(f"No matching files found for pattern: {file_pattern}")
            st.error(f"❌ No match maker files found for request ID: {ai_request_id}")
            return None
        
        logger.info(f"Found latest match maker file: {latest_file}")
        
        # Get the file
        response = s3.get_object(Bucket=MATCH_MAKING_BUCKET, Key=latest_file)
        match_data = json.loads(response['Body'].read().decode('utf-8'))
        
        matches = match_data.get('matches', [])
        if matches:
            logger.info(f"Extracted {len(matches)} matches from S3")
            return {
                'selected_match': matches[0],  # Assume first is selected
                'matches': matches
            }
        
        logger.warning(f"No matches found in file: {latest_file}")
        st.error(f"❌ No matches found in match maker data for request ID: {ai_request_id}")
        return None
    except Exception as e:
        error_msg = f"Error fetching match maker data: {str(e)}"
        logger.error(error_msg)
        st.error(f"❌ {error_msg}")
        
        # Provide specific guidance for common errors
        if "NoSuchBucket" in str(e):
            st.error(f"🔍 The S3 bucket '{MATCH_MAKING_BUCKET}' does not exist or is not accessible.")
            st.info("💡 Please check that the bucket name is correct and your IAM permissions include 's3:ListBucket' and 's3:GetObject'.")
        elif "AccessDenied" in str(e):
            st.error("🔒 Access denied to S3. Your AWS credentials may not have the necessary permissions.")
            st.info("💡 Please ensure your IAM role/user has permissions for S3 operations.")
        
        return None

def get_survey_analysis_data(ai_request_id: str, form_id: str = None) -> Dict:
    """
    Get survey analysis data from DynamoDB using query on composite key.

    IMPORTANT: Unlike other agents, this uses query() not get_item() because
    the sort key is composite: FORM{id}#SURVEY_ANALYSIS

    Args:
        ai_request_id: The AI response ID (partition key, same as ai_request_id)
        form_id: Optional form ID for filtering specific form analysis

    Returns:
        Dictionary with batch_insights, clusters, sentiment_analysis, executive_summary
    """
    try:
        dynamodb, _, _ = init_aws_clients()
        if not dynamodb:
            st.error("❌ Cannot access DynamoDB. AWS clients not initialized properly.")
            return None

        table = dynamodb.Table(FORM_RESPONSES_TABLE)
        logger.info(f"Attempting to query survey analysis for request ID: {ai_request_id}")

        # Build query - use contains for the composite key suffix
        from boto3.dynamodb.conditions import Key

        if form_id:
            # Query for specific form
            composite_key = f'FORM{form_id}#SURVEY_ANALYSIS'
            response = table.query(
                KeyConditionExpression=Key('ai_response_id').eq(ai_request_id) & Key('composite').eq(composite_key)
            )
        else:
            # Query all SURVEY_ANALYSIS entries for this request
            # Note: DynamoDB doesn't support 'contains' in KeyConditionExpression
            # We need to use begins_with or scan. Let's use scan with filter for now
            response = table.query(
                KeyConditionExpression=Key('ai_response_id').eq(ai_request_id),
                FilterExpression='contains(composite, :suffix)',
                ExpressionAttributeValues={':suffix': '#SURVEY_ANALYSIS'}
            )

        if response.get('Items'):
            # Get the first item (or most recent if multiple)
            item = response['Items'][0]
            logger.info(f"Found survey analysis for request ID: {ai_request_id}")

            # Parse the analysis JSON
            analysis_str = item.get('analysis', '{}')
            if isinstance(analysis_str, str):
                analysis = json.loads(analysis_str)
            else:
                analysis = analysis_str

            # Extract result structure
            result = analysis.get('result', {})
            batch_insights = result.get('batch_insights', {})
            clusters = result.get('clusters', [])
            sentiment_analysis = result.get('sentiment_analysis', {})
            executive_summary = result.get('executive_summary', {})

            logger.info(f"Extracted {len(clusters)} customer segments from survey analysis")

            return {
                'batch_insights': batch_insights,
                'clusters': clusters,
                'sentiment_analysis': sentiment_analysis,
                'executive_summary': executive_summary,
                'form_id': item.get('composite', '').split('#')[0].replace('FORM', ''),
                'timestamp': item.get('timestamp', '')
            }
        else:
            logger.warning(f"No survey analysis found for request ID: {ai_request_id}")
            st.error(f"❌ No survey analysis data found for request ID: {ai_request_id}")
            if form_id:
                st.info(f"💡 Searched for form_id: {form_id}")
            return None

    except Exception as e:
        error_msg = f"Error fetching survey analysis data: {str(e)}"
        logger.error(error_msg)
        st.error(f"❌ {error_msg}")

        if "ResourceNotFoundException" in str(e):
            st.error("🔍 The FORM_RESPONSES_TABLE might not exist or your credentials don't have access.")
            st.info("💡 Please check that the table name is correct and your IAM permissions include 'dynamodb:Query'.")

        return None

def get_meeting_summary_data(ai_request_id: str, meeting_id: str, user_id: str, persona_type: str) -> Dict:
    """
    Get meeting summary data from DynamoDB using query on composite key.

    IMPORTANT: Uses query() because sort key is composite: USER{user_id}#MEETING{meeting_id}

    Args:
        ai_request_id: The AI request ID (partition key)
        meeting_id: Meeting identifier
        user_id: User identifier
        persona_type: Either 'SME' or 'FOUNDER'

    Returns:
        Dictionary with sme_bullet_points or founder_bullet_points, status, timestamp
    """
    try:
        dynamodb, _, _ = init_aws_clients()
        if not dynamodb:
            st.error("❌ Cannot access DynamoDB. AWS clients not initialized properly.")
            return None

        table = dynamodb.Table(MEETING_SUMMARY_TABLE)
        logger.info(f"Attempting to query meeting summary for request ID: {ai_request_id}, meeting: {meeting_id}")

        # Build composite sort key
        from boto3.dynamodb.conditions import Key
        composite_key = f'USER{user_id}#MEETING{meeting_id}'

        # Execute query
        response = table.query(
            KeyConditionExpression=Key('ai_request_id').eq(ai_request_id) & Key('composite').eq(composite_key)
        )

        if response.get('Items'):
            item = response['Items'][0]
            logger.info(f"Found meeting summary for request ID: {ai_request_id}")

            # Parse the summary JSON
            summary_str = item.get('summary', '{}')
            if isinstance(summary_str, str):
                summary = json.loads(summary_str)
            else:
                summary = summary_str

            # Extract persona-specific bullet points
            persona_type_upper = persona_type.upper()
            if persona_type_upper == 'SME':
                bullet_points = summary.get('sme_bullet_points', '')
            elif persona_type_upper == 'FOUNDER':
                bullet_points = summary.get('founder_bullet_points', '')
            else:
                logger.warning(f"Unknown persona type: {persona_type}")
                bullet_points = ''

            logger.info(f"Extracted {persona_type} meeting summary")

            return {
                'bullet_points': bullet_points,
                'persona_type': persona_type_upper,
                'status': summary.get('status', 'unknown'),
                'generated_at': summary.get('generated_at', ''),
                'timestamp': summary.get('timestamp', ''),
                'meeting_id': meeting_id,
                'user_id': user_id
            }
        else:
            logger.warning(f"No meeting summary found for request ID: {ai_request_id}, meeting: {meeting_id}")
            st.error(f"❌ No meeting summary data found")
            st.info(f"💡 Searched for: ai_request_id={ai_request_id}, user={user_id}, meeting={meeting_id}")
            return None

    except Exception as e:
        error_msg = f"Error fetching meeting summary data: {str(e)}"
        logger.error(error_msg)
        st.error(f"❌ {error_msg}")

        if "ResourceNotFoundException" in str(e):
            st.error("🔍 The MEETING_SUMMARY_TABLE might not exist or your credentials don't have access.")
            st.info("💡 Please check that the table name is correct and your IAM permissions include 'dynamodb:Query'.")

        return None

def get_transcript_summary_data(ai_request_id: str, meeting_id: str, user_id: str, persona_type: str) -> Dict:
    """
    Get transcript summary data from DynamoDB (same table as meeting summary).

    IMPORTANT: Uses query() because sort key is composite: USER{user_id}#MEETING{meeting_id}

    Args:
        ai_request_id: The AI request ID (partition key)
        meeting_id: Meeting identifier
        user_id: User identifier
        persona_type: One of 'peer', 'sme', or 'founder'

    Returns:
        Dictionary with insights array, quotes array, persona_type
    """
    try:
        dynamodb, _, _ = init_aws_clients()
        if not dynamodb:
            st.error("❌ Cannot access DynamoDB. AWS clients not initialized properly.")
            return None

        table = dynamodb.Table(MEETING_SUMMARY_TABLE)
        logger.info(f"Attempting to query transcript summary for request ID: {ai_request_id}, meeting: {meeting_id}")

        # Build composite sort key
        from boto3.dynamodb.conditions import Key
        composite_key = f'USER{user_id}#MEETING{meeting_id}'

        # Execute query
        response = table.query(
            KeyConditionExpression=Key('ai_request_id').eq(ai_request_id) & Key('composite').eq(composite_key)
        )

        if response.get('Items'):
            item = response['Items'][0]
            logger.info(f"Found transcript summary for request ID: {ai_request_id}")

            # Parse the transcript_summary JSON
            transcript_str = item.get('transcript_summary', '{}')
            if isinstance(transcript_str, str):
                transcript = json.loads(transcript_str)
            else:
                transcript = transcript_str

            # Extract insights and quotes
            insights = transcript.get('insights', [])
            quotes = transcript.get('quotes', [])
            detected_persona = transcript.get('persona_type', persona_type)

            logger.info(f"Extracted {len(insights)} insights and {len(quotes)} quotes from transcript")

            return {
                'insights': insights,
                'quotes': quotes,
                'persona_type': detected_persona,
                'meeting_id': meeting_id,
                'user_id': user_id,
                'timestamp': item.get('timestamp', '')
            }
        else:
            logger.warning(f"No transcript summary found for request ID: {ai_request_id}, meeting: {meeting_id}")
            st.error(f"❌ No transcript summary data found")
            st.info(f"💡 Searched for: ai_request_id={ai_request_id}, user={user_id}, meeting={meeting_id}")
            return None

    except Exception as e:
        error_msg = f"Error fetching transcript summary data: {str(e)}"
        logger.error(error_msg)
        st.error(f"❌ {error_msg}")

        if "ResourceNotFoundException" in str(e):
            st.error("🔍 The MEETING_SUMMARY_TABLE might not exist or your credentials don't have access.")
            st.info("💡 Please check that the table name is correct and your IAM permissions include 'dynamodb:Query'.")

        return None

def convert_floats_to_decimal(obj):
    """
    Recursively convert all float values in a data structure to Decimal for DynamoDB compatibility.

    DynamoDB doesn't support Python's native float type. This function converts:
    - float -> Decimal
    - dict -> recursively process all values
    - list -> recursively process all items
    - other types -> return as-is

    Args:
        obj: Any Python object (dict, list, float, int, str, etc.)

    Returns:
        The same structure with all floats converted to Decimal
    """
    if isinstance(obj, float):
        return Decimal(str(obj))
    elif isinstance(obj, dict):
        return {key: convert_floats_to_decimal(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_floats_to_decimal(item) for item in obj]
    else:
        return obj

def save_feedback_to_dynamodb(stage: str, ai_request_id: str, responses: Dict,
                               questions: List[Dict], analysis_summary: str,
                               additional_metadata: Dict = None) -> bool:
    """
    Save feedback responses to DynamoDB feedback table.

    Stores data in a column named after the stage (e.g., 'idea_capture', 'survey_analysis').
    If the ai_request_id already exists, updates that stage's column.

    Table structure:
    - Partition key: ai_request_id
    - Columns: idea_capture, lens_selector, survey_generator, 360_report,
               social_lens, match_maker, survey_analysis, meeting_summary,
               transcript_summary (each contains feedback data for that stage)

    Args:
        stage: The AI agent stage name
        ai_request_id: The AI request ID (partition key)
        responses: Dictionary of question_id -> selected_value
        questions: List of question objects with metadata
        analysis_summary: The AI-generated analysis summary
        additional_metadata: Optional dict with stage-specific metadata

    Returns:
        True if saved successfully, False otherwise
    """
    try:
        dynamodb, _, _ = init_aws_clients()
        if not dynamodb:
            st.error("❌ Cannot access DynamoDB for saving feedback")
            return False

        table = dynamodb.Table(FEEDBACK_TABLE)
        timestamp = datetime.now().isoformat()

        # Enrich responses with question metadata
        enriched_responses = []
        for question in questions:
            q_id = question.get('id')
            if q_id in responses:
                enriched_responses.append({
                    'question_id': q_id,
                    'question_text': question.get('text', ''),
                    'focus_area': question.get('focus_area', ''),
                    'selected_value': responses[q_id],
                    'selected_label': next(
                        (opt['label'] for opt in question.get('options', [])
                         if opt['value'] == responses[q_id]),
                        ''
                    ),
                    'rl_value': question.get('rl_value', 0.0)
                })

        # Build the stage-specific feedback data
        stage_data = {
            'responses': enriched_responses,
            'analysis_summary': analysis_summary,
            'timestamp': timestamp,
            'num_questions': len(questions),
            'num_responses': len(responses)
        }

        # Add metadata if provided
        if additional_metadata:
            stage_data['metadata'] = additional_metadata

        # Convert all float values to Decimal for DynamoDB compatibility
        stage_data = convert_floats_to_decimal(stage_data)

        # Update the item - use UpdateExpression to set/update the specific stage column
        # This allows multiple stages to be stored in the same item
        table.update_item(
            Key={'ai_request_id': ai_request_id},
            UpdateExpression=f'SET #{stage} = :stage_data',
            ExpressionAttributeNames={
                f'#{stage}': stage  # Use expression attribute names to avoid reserved words
            },
            ExpressionAttributeValues={
                ':stage_data': stage_data
            }
        )

        logger.info(f"✓ Saved feedback for stage '{stage}' to column '{stage}', request ID: {ai_request_id}")
        return True

    except Exception as e:
        error_msg = f"Error saving feedback to DynamoDB: {str(e)}"
        logger.error(error_msg)
        st.error(f"❌ {error_msg}")
        return False

def generate_feedback_questions(stage: str, ai_request_id: str, persona_type: str = None) -> List[Dict]:
    """
    Generate feedback questions using two-stage AI generation:
    Stage 1: Generate evaluation questions with focus areas
    Stage 2: Dynamically generate contextual options for each question
    """

    # Stage-specific data retrieval and prompt building
    data = None
    prompt = None
    agent_name = ""
    context_summary = ""

    if stage == 'idea_capture':
        data = get_idea_capture_data(ai_request_id)
        if data:
            prompt = build_idea_capture_prompt(
                data['burning_issues'],
                data['domain'],
                data['input_metadata']
            )
            agent_name = "Idea Capture Agent"
            issues_titles = [issue.get('title', '') for issue in data['burning_issues'][:2]]
            context_summary = f"Identified burning issues in {data['domain']}: {', '.join(issues_titles)}"

    elif stage == 'lens_selector':
        data = get_lens_selector_data(ai_request_id)
        if data:
            prompt = build_lens_selector_prompt(
                data['results'],
                data['domain'],
                data['burning_issues_summary']
            )
            agent_name = "Lens Selector Agent"
            first_lens = data['results'][0]['lens'] if data['results'] else 'Unknown'
            context_summary = f"Recommended {first_lens} as primary validation lens for {data['domain']}"

    elif stage == 'survey_generator':
        data = get_survey_generator_data(ai_request_id)
        if data:
            prompt = build_survey_generator_prompt(
                data['questions'],
                data['domain'],
                data['burning_issues']
            )
            agent_name = "Survey Generator Agent"
            context_summary = f"Generated {len(data['questions'])} survey questions for {data['domain']} validation"

    elif stage == '360_report':
        data = get_360_report_data(ai_request_id)
        if data:
            prompt = build_360_report_prompt(
                data['verdict'],
                data['composite_score'],
                data['confidence']
            )
            agent_name = "360 Report Agent"
            context_summary = f"Verdict: {data['verdict']}, Score: {data['composite_score']}/100"

    elif stage == 'social_lens':
        data = get_social_lens_data(ai_request_id)
        if data:
            prompt = build_social_lens_prompt(
                data['buzz_score'],
                data['buzz_level'],
                data['sources']
            )
            agent_name = "Social Synth Agent"
            context_summary = f"Buzz score: {data['buzz_score']}/100, Level: {data['buzz_level']}"

    elif stage == 'match_maker':
        if persona_type:
            data = get_match_maker_data(ai_request_id, persona_type)
            if data:
                match_score = data['selected_match'].get('match_score', 0)
                prompt = build_match_maker_prompt(
                    data['selected_match'],
                    match_score
                )
                agent_name = "Match-Maker Agent"
                match_name = data['selected_match'].get('name', 'Unknown')
                context_summary = f"Matched {match_name} ({persona_type}) with {match_score:.0%} confidence"

    elif stage == 'survey_analysis':
        # Check if form_id is provided (optional)
        form_id = st.session_state.get('form_id', None)
        data = get_survey_analysis_data(ai_request_id, form_id)
        if data:
            prompt = build_survey_analysis_prompt(
                data['batch_insights'],
                data['clusters'],
                data['sentiment_analysis'],
                data['executive_summary']
            )
            agent_name = "Survey Analysis Agent"
            num_segments = len(data['clusters'])
            primary_segment = data['executive_summary'].get('primary_segment', 'Unknown')
            context_summary = f"Identified {num_segments} customer segments, primary: {primary_segment}"

    elif stage == 'meeting_summary':
        # Require meeting_id, user_id, and persona_type
        meeting_id = st.session_state.get('meeting_id')
        user_id = st.session_state.get('user_id')

        if not all([meeting_id, user_id, persona_type]):
            st.error("❌ Missing required inputs: meeting_id, user_id, or persona_type")
            return {"questions": [], "analysis_summary": "Missing required inputs"}

        data = get_meeting_summary_data(ai_request_id, meeting_id, user_id, persona_type)
        if data:
            prompt = build_meeting_summary_prompt(
                data['bullet_points'],
                data['persona_type'],
                data['meeting_id']
            )
            agent_name = "Meeting Summary Agent"
            context_summary = f"{data['persona_type']} meeting prep for meeting {meeting_id}"

    elif stage == 'transcript_summary':
        # Require meeting_id, user_id, and persona_type
        meeting_id = st.session_state.get('meeting_id')
        user_id = st.session_state.get('user_id')

        if not all([meeting_id, user_id, persona_type]):
            st.error("❌ Missing required inputs: meeting_id, user_id, or persona_type")
            return {"questions": [], "analysis_summary": "Missing required inputs"}

        data = get_transcript_summary_data(ai_request_id, meeting_id, user_id, persona_type)
        if data:
            prompt = build_transcript_summary_prompt(
                data['insights'],
                data['quotes'],
                data['persona_type']
            )
            agent_name = "Transcript Summary Agent"
            num_insights = len(data['insights'])
            context_summary = f"Extracted {num_insights} insights from {data['persona_type']} conversation"

    # If no data or prompt could be built, return empty dict
    if not prompt or not data:
        return {"questions": [], "analysis_summary": "No data available for this stage and request ID."}

    # STAGE 1: Generate evaluation questions WITH analysis summary AND options
    response = invoke_bedrock(prompt, max_tokens=1500)
    if not response:
        st.error("Failed to generate questions from AI")
        return {"questions": [], "analysis_summary": ""}

    try:
        # Extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            questions_data = json.loads(json_match.group())
        else:
            questions_data = json.loads(response)

        questions = questions_data.get('questions', [])
        analysis_summary = questions_data.get('analysis_summary', '')

        if not questions:
            st.error("No questions generated")
            return {"questions": [], "analysis_summary": analysis_summary}

        # Rename 'focus' to 'focus_area' if needed and ensure options exist
        enriched_questions = []
        for question in questions:
            # Handle focus_area naming
            if 'focus' in question and 'focus_area' not in question:
                question['focus_area'] = question['focus']
                del question['focus']

            # Ensure options exist (they should be generated by the prompt)
            if 'options' not in question or not question['options']:
                # Fallback: generate options if missing
                focus_area = question.get('focus_area', 'quality')
                logger.warning(f"Options missing for question {question.get('id')}, using fallback")
                question['options'] = get_fallback_options(focus_area)

            enriched_questions.append(question)
            logger.info(f"Question {question.get('id', 'unknown')}: {len(question['options'])} options")

        logger.info(f"Generated {len(enriched_questions)} questions with analysis summary")

        return {
            "questions": enriched_questions,
            "analysis_summary": analysis_summary
        }

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response: {str(e)}")
        st.error(f"Failed to parse AI response: {str(e)}")
        return {"questions": [], "analysis_summary": ""}
    except Exception as e:
        logger.error(f"Error in generate_feedback_questions: {str(e)}")
        st.error(f"Error generating questions: {str(e)}")
        return {"questions": [], "analysis_summary": ""}

# Streamlit UI
st.set_page_config(
    page_title="AI Feedback System",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI-Generated Dynamic Feedback System")
st.markdown("---")

# Sidebar for inputs
with st.sidebar:
    st.header("Configuration")
    
    # Stage selection
    stage = st.selectbox(
        "Select AI Agent Stage",
        ["idea_capture", "lens_selector", "survey_generator", "360_report", "social_lens",
         "match_maker", "survey_analysis", "meeting_summary", "transcript_summary"],
        index=0,
        key="stage_select"
    )
    
    # AI Request ID
    ai_request_id = st.text_input(
        "AI Request ID",
        value="ccd233f0-eb97-4783-b43b-2a2a284d51bc",
        key="ai_request_id_input"
    )
    
    # Persona type for match_maker
    persona_type = None
    if stage == "match_maker":
        persona_type = st.selectbox(
            "Persona Type",
            ["SME", "FOUNDER", "RESPONDENT"],
            key="persona_type_select"
        )

    # Additional inputs for survey_analysis
    if stage == "survey_analysis":
        form_id = st.text_input(
            "Form ID (optional)",
            value="",
            key="form_id_input",
            help="Leave blank to analyze all forms for this request"
        )
        if form_id:
            st.session_state.form_id = form_id

    # Additional inputs for meeting_summary and transcript_summary
    if stage in ["meeting_summary", "transcript_summary"]:
        meeting_id = st.text_input(
            "Meeting ID",
            value="",
            key="meeting_id_input",
            help="Required: Meeting identifier"
        )
        st.session_state.meeting_id = meeting_id

        user_id = st.text_input(
            "User ID",
            value="",
            key="user_id_input",
            help="Required: User identifier"
        )
        st.session_state.user_id = user_id

        if stage == "meeting_summary":
            persona_type = st.selectbox(
                "Persona Type",
                ["SME", "FOUNDER"],
                key="persona_type_meeting_select",
                help="Select the persona type for meeting summary"
            )
        else:  # transcript_summary
            persona_type = st.selectbox(
                "Persona Type",
                ["peer", "sme", "founder"],
                key="persona_type_transcript_select",
                help="Select the persona type for transcript analysis"
            )

        # Validation warning
        if not meeting_id or not user_id:
            st.warning("⚠️ Meeting ID and User ID are required for this stage")

    # Generate button
    generate_button = st.button(
        "Generate Feedback Questions",
        type="primary",
        use_container_width=True
    )

# Main content area
col1, col2 = st.columns([2, 1])

# Add AWS status check at the top
with st.expander("🔍 AWS System Status", expanded=False):
    st.write("Checking AWS resource availability...")
    
    try:
        dynamodb, s3, bedrock_runtime = init_aws_clients()
        if dynamodb and s3 and bedrock_runtime:
            table_names = [IDEAS_TABLE, FINALIZE_REPORT_TABLE, SOCIAL_LENS_TABLE,
                          FORM_RESPONSES_TABLE, MEETING_SUMMARY_TABLE, FEEDBACK_TABLE]
            resource_status = check_aws_resources_exist(dynamodb, s3, bedrock_runtime, table_names, MATCH_MAKING_BUCKET)
            
            status_message = get_resource_status_message(resource_status)
            if resource_status['all_resources_available']:
                st.success(status_message)
            else:
                st.error(status_message)
                
            # Show detailed status
            st.subheader("Detailed Status")
            
            # DynamoDB tables
            st.write("**DynamoDB Tables:**")
            for table_name, exists in resource_status['dynamodb_tables'].items():
                status_icon = "✅" if exists else "❌"
                st.write(f"{status_icon} {table_name}: {'Accessible' if exists else 'Not found or inaccessible'}")
            
            # S3 bucket
            st.write("**S3 Bucket:**")
            bucket_status = "✅" if resource_status['s3_bucket'] else "❌"
            st.write(f"{bucket_status} {MATCH_MAKING_BUCKET}: {'Accessible' if resource_status['s3_bucket'] else 'Not found or inaccessible'}")
            
            # Bedrock access
            st.write("**Bedrock Runtime:**")
            bedrock_status = "✅" if resource_status['bedrock_access'] else "❌"
            st.write(f"{bedrock_status} Bedrock Runtime: {'Accessible' if resource_status['bedrock_access'] else 'Not accessible'}")
        else:
            st.error("❌ Failed to initialize AWS clients. Please check your AWS configuration.")
    except Exception as e:
        st.error(f"❌ Error checking AWS status: {str(e)}")

with col1:
    st.header("Feedback Questions")
    
    if generate_button:
        with st.spinner("Generating feedback questions..."):
            try:
                result = generate_feedback_questions(stage, ai_request_id, persona_type)

                # Handle the case where result might not be a dict
                if not isinstance(result, dict):
                    st.error(f"Unexpected result type: {type(result)}. Expected dict.")
                    st.error(f"Result value: {result}")
                    result = {"questions": [], "analysis_summary": "Error: Invalid result format"}

                if result.get('questions'):
                    st.session_state.questions = result['questions']
                    st.session_state.analysis_summary = result.get('analysis_summary', '')
                    st.session_state.current_stage = stage
                    st.session_state.ai_request_id = ai_request_id
                    st.success(f"Generated {len(result['questions'])} feedback questions!")
                else:
                    error_msg = result.get('analysis_summary', 'Failed to generate questions')
                    st.error(f"No questions generated: {error_msg}")
            except Exception as e:
                st.error(f"Error generating questions: {str(e)}")
                st.exception(e)

    # Display questions
    if st.session_state.questions:
        # Show analysis summary at top if available
        if st.session_state.get('analysis_summary'):
            st.info(f"**Quality Analysis:** {st.session_state['analysis_summary']}")
            st.markdown("---")

        st.subheader(f"Questions for {stage.replace('_', ' ').title()}")

        for i, question in enumerate(st.session_state.questions):
            with st.container():
                st.write(f"**Q{i+1}: {question['text']}**")

                # Display focus area
                if 'focus_area' in question:
                    focus_display = question['focus_area'].replace('_', ' ').title()
                    st.caption(f"📊 Focus: {focus_display}")

                # Create radio buttons for options
                options = {opt['value']: opt['label'] for opt in question['options']}
                selected_option = st.radio(
                    f"Select an option for Q{i+1}:",
                    list(options.keys()),
                    format_func=lambda x: options[x],
                    key=f"q_{question['id']}"
                )

                # Store response
                st.session_state.responses[question['id']] = selected_option

                st.markdown("---")
        
        # Submit button
        if st.button("Submit Feedback", type="primary", use_container_width=True):
            # Prepare stage-specific metadata
            metadata = {}

            if st.session_state.current_stage == 'survey_analysis':
                if st.session_state.get('form_id'):
                    metadata['form_id'] = st.session_state.form_id

            elif st.session_state.current_stage in ['meeting_summary', 'transcript_summary']:
                if st.session_state.get('meeting_id'):
                    metadata['meeting_id'] = st.session_state.meeting_id
                if st.session_state.get('user_id'):
                    metadata['user_id'] = st.session_state.user_id
                if st.session_state.get('persona_type'):
                    metadata['persona_type'] = st.session_state.persona_type

            elif st.session_state.current_stage == 'match_maker':
                if st.session_state.get('persona_type'):
                    metadata['persona_type'] = st.session_state.persona_type

            # Save to DynamoDB
            success = save_feedback_to_dynamodb(
                stage=st.session_state.current_stage,
                ai_request_id=st.session_state.ai_request_id,
                responses=st.session_state.responses,
                questions=st.session_state.questions,
                analysis_summary=st.session_state.get('analysis_summary', ''),
                additional_metadata=metadata if metadata else None
            )

            if success:
                st.success(f"✅ Feedback for '{st.session_state.current_stage}' submitted and saved!")

                # Display what was saved
                st.subheader("Saved Feedback:")
                st.json({
                    'ai_request_id': st.session_state.ai_request_id,
                    'stage_column': st.session_state.current_stage,
                    'num_responses': len(st.session_state.responses),
                    'metadata': metadata,
                    'timestamp': datetime.now().isoformat()
                })

                # Optionally clear responses for next evaluation
                if st.checkbox("Clear responses for next agent evaluation?"):
                    st.session_state.responses = {}
                    st.rerun()
            else:
                st.error("❌ Failed to save feedback. Please try again.")

with col2:
    st.header("System Info")
    
    st.subheader("Current Configuration")
    st.write(f"**Stage:** {stage}")
    st.write(f"**Request ID:** {ai_request_id}")
    if persona_type:
        st.write(f"**Persona Type:** {persona_type}")

    # Display new stage-specific inputs
    if stage == "survey_analysis" and st.session_state.get('form_id'):
        st.write(f"**Form ID:** {st.session_state.form_id}")

    if stage in ["meeting_summary", "transcript_summary"]:
        if st.session_state.get('meeting_id'):
            st.write(f"**Meeting ID:** {st.session_state.meeting_id}")
        if st.session_state.get('user_id'):
            st.write(f"**User ID:** {st.session_state.user_id}")

    st.subheader("Model Configuration")
    st.write(f"**Bedrock Model:** {BEDROCK_MODEL_ID}")
    st.write(f"**Ideas Table:** {IDEAS_TABLE}")
    st.write(f"**Report Table:** {FINALIZE_REPORT_TABLE}")
    st.write(f"**Social Table:** {SOCIAL_LENS_TABLE}")
    st.write(f"**S3 Bucket:** {MATCH_MAKING_BUCKET}")
    st.write(f"**Form Responses Table:** {FORM_RESPONSES_TABLE}")
    st.write(f"**Meeting Summary Table:** {MEETING_SUMMARY_TABLE}")
    st.write(f"**Feedback Table:** {FEEDBACK_TABLE}")
    
    st.subheader("Session State")
    if st.session_state.current_stage:
        st.write(f"**Current Stage:** {st.session_state.current_stage}")
    if st.session_state.ai_request_id:
        st.write(f"**Request ID:** {st.session_state.ai_request_id}")
    st.write(f"**Questions Generated:** {len(st.session_state.questions)}")
    st.write(f"**Responses Collected:** {len(st.session_state.responses)}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>AI-Generated Dynamic Feedback System • Powered by AWS Bedrock • Streamlit Interface</p>
    </div>
    """,
    unsafe_allow_html=True
)