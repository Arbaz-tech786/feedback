# Streamlit App Update Guide

This guide provides the specific changes needed to update the Streamlit app with improved prompts that generate dynamic option values.

## Key Issues to Address

1. **Hardcoded option values**: The current app uses predefined values like "excellent", "good", "fair", "poor" which are not context-specific.

2. **Generic prompts**: The current prompts are too generic and don't provide enough context for generating high-quality, specific feedback questions.

3. **Limited context analysis**: The app doesn't analyze the quality of agent outputs before generating questions.

## Required Changes

### 1. Add Quality Analysis Functions

Add these functions after the `extract_domain` function (around line 104):

```python
def analyze_burning_issues_quality(burning_issues: List[Dict]) -> Dict:
    """Analyze the quality of burning issues"""
    quality = {}
    
    for i, issue in enumerate(burning_issues, 1):
        score = 0
        reasons = []
        
        # Check hypothesis quality
        hypothesis = issue.get('hypothesis', '')
        if len(hypothesis) > 50 and 'will' in hypothesis.lower():
            score += 2
            reasons.append("Testable hypothesis")
        
        # Check measurement approach
        measurement = issue.get('how_to_measure', '')
        if measurement and 'comparative' in measurement.lower():
            score += 2
            reasons.append("Clear measurement approach")
        
        # Check success threshold
        threshold = issue.get('success_threshold', '')
        if threshold and '%' in threshold:
            score += 1
            reasons.append("Quantifiable success criteria")
        
        # Assign quality rating
        if score >= 4:
            quality[f'issue_{i}'] = "High quality - " + ", ".join(reasons)
        elif score >= 2:
            quality[f'issue_{i}'] = "Medium quality - " + ", ".join(reasons)
        else:
            quality[f'issue_{i}'] = "Low quality - Needs improvement"
    
    return quality

def analyze_lens_selection_quality(results: List[Dict], burning_issues: List[str]) -> Dict:
    """Analyze the quality of lens selection"""
    quality = {}
    
    # Check ranking coherence
    top_lens = [r.get('lens', '') for r in results[:3]]
    if 'Survey' in top_lens:
        quality['ranking_coherence'] = "Good - Survey prioritized for validation"
    else:
        quality['ranking_coherence'] = "Questionable - Survey not prioritized"
    
    # Check confidence justification
    avg_confidence = sum(r.get('confidence', 0) for r in results[:3]) / 3
    if avg_confidence > 0.7:
        quality['confidence_justification'] = "High confidence with good justification"
    else:
        quality['confidence_justification'] = "Low confidence - needs better justification"
    
    # Check stage appropriateness
    stages = [r.get('stageRelevance', 0) for r in results[:3]]
    avg_stage = sum(stages) / len(stages)
    if avg_stage > 0.7:
        quality['stage_appropriateness'] = "High stage relevance"
    else:
        quality['stage_appropriateness'] = "Low stage relevance - reconsider"
    
    return quality

def analyze_survey_quality(questions: List[Dict], burning_issues: List[Dict]) -> Dict:
    """Analyze the quality of survey questions"""
    quality = {}
    
    # Check question clarity
    clear_questions = sum(1 for q in questions if len(q.get('text', '')) < 100)
    quality['question_clarity'] = f"{clear_questions}/{len(questions)} questions clear"
    
    # Check burning issue alignment
    aligned_questions = sum(1 for q in questions if q.get('burning_problem_reference'))
    quality['bi_alignment'] = f"{aligned_questions}/{len(questions)} questions aligned to BIs"
    
    # Check question type variety
    question_types = set(q.get('type', '') for q in questions)
    quality['type_variety'] = f"{len(question_types)} different question types"
    
    # Check survey flow
    guardrail_questions = sum(1 for q in questions if 'guardrail' in q.get('bucket', ''))
    quality['flow_logic'] = f"{guardrail_questions} guardrail questions for flow"
    
    return quality

def analyze_report_quality(report_data: Dict) -> Dict:
    """Analyze the quality of 360 report"""
    quality = {}
    
    verdict = report_data.get('verdict', {}).get('decision', '')
    composite_score = report_data.get('verdict', {}).get('compositeScore', 0)
    
    # Check verdict justification
    if composite_score > 50 and verdict in ['Pivot', 'Proceed']:
        quality['verdict_justification'] = "Verdict aligns with score"
    else:
        quality['verdict_justification'] = "Verdict may not align with score"
    
    # Check risk identification
    risks = report_data.get('topRisks', [])
    quality['risk_identification'] = f"{len(risks)} key risks identified"
    
    # Check action quality
    actions = report_data.get('topNextActions', [])
    quality['action_quality'] = f"{len(actions)} actionable next steps"
    
    # Check overall coherence
    if len(risks) > 0 and len(actions) > 0:
        quality['overall_coherence'] = "Good balance of risks and actions"
    else:
        quality['overall_coherence'] = "Missing risks or actions"
    
    return quality

def analyze_social_quality(social_data: Dict) -> Dict:
    """Analyze the quality of social analysis"""
    quality = {}
    
    buzz_score = social_data.get('overall_buzz_score', 0)
    sources = social_data.get('sources', [])
    talking_points = social_data.get('main_talking_points', [])
    
    # Check buzz scoring
    if buzz_score > 70:
        quality['buzz_scoring'] = "High buzz detected"
    elif buzz_score > 40:
        quality['buzz_scoring'] = "Moderate buzz detected"
    else:
        quality['buzz_scoring'] = "Low buzz detected"
    
    # Check source relevance
    quality['source_relevance'] = f"{len(sources)} sources analyzed"
    
    # Check sentiment analysis
    sentiment_groups = social_data.get('group_sentiment', [])
    quality['sentiment_analysis'] = f"Sentiment analyzed for {len(sentiment_groups)} groups"
    
    # Check trend identification
    trending = social_data.get('topics_getting_hotter', [])
    quality['trend_identification'] = f"{len(trending)} trending topics identified"
    
    # Check overall insight
    if len(talking_points) >= 3 and len(trending) >= 3:
        quality['overall_insight'] = "Comprehensive social analysis"
    else:
        quality['overall_insight'] = "Limited social insights"
    
    return quality

def analyze_matching_quality(selected_match: Dict, all_matches: List[Dict], persona_type: str) -> Dict:
    """Analyze the quality of expert matching"""
    quality = {}
    
    match_score = selected_match.get('match_score', 0)
    expertise = selected_match.get('expertise', [])
    
    # Check score justification
    if match_score > 0.8:
        quality['score_justification'] = "High match score"
    elif match_score > 0.6:
        quality['score_justification'] = "Moderate match score"
    else:
        quality['score_justification'] = "Low match score - questionable"
    
    # Check expertise relevance
    quality['expertise_relevance'] = f"{len(expertise)} relevant expertise areas"
    
    # Check persona fit
    if persona_type.lower() in ['sme', 'founder', 'respondent']:
        quality['persona_fit'] = f"Matched for {persona_type} persona"
    else:
        quality['persona_fit'] = "Unknown persona fit"
    
    # Check overall quality
    if match_score > 0.7 and len(expertise) > 2:
        quality['overall_quality'] = "High quality match"
    else:
        quality['overall_quality'] = "Questionable match quality"
    
    # Check alternative consideration
    if len(all_matches) > 1:
        second_best = all_matches[1].get('match_score', 0)
        if match_score - second_best > 0.2:
            quality['alternative_consideration'] = "Clear winner selected"
        else:
            quality['alternative_consideration'] = "Close alternatives available"
    else:
        quality['alternative_consideration'] = "No alternatives to consider"
    
    return quality

def analyze_follow_up_quality(follow_up_answers: List[Dict]) -> str:
    """Analyze the quality of follow-up answers"""
    if not follow_up_answers:
        return "No follow-up answers provided"
    
    quality_answers = sum(1 for answer in follow_up_answers 
                         if len(answer.get('answer', '')) > 10 
                         and answer.get('answer', '').lower() not in ['okay', 'yes', 'no'])
    
    if quality_answers == len(follow_up_answers):
        return "High quality - detailed answers provided"
    elif quality_answers > len(follow_up_answers) / 2:
        return "Medium quality - some detailed answers"
    else:
        return "Low quality - mostly minimal answers"
```

### 2. Update Data Fetching Functions

Update the `get_idea_capture_data` function (around line 441) to include more data:

```python
def get_idea_capture_data(ai_request_id: str) -> Dict:
    """Get idea capture data from DynamoDB"""
    try:
        dynamodb, _, _ = init_aws_clients()
        if not dynamodb:
            return None
        
        table = dynamodb.Table(IDEAS_TABLE)
        response = table.get_item(Key={'ai_request_id': ai_request_id})
        
        if 'Item' in response:
            item = response['Item']
            burning_issues = item.get('burning_issues', [])
            input_metadata = item.get('input_metadata', {})
            domain = extract_domain(input_metadata)
            
            return {
                'burning_issues': burning_issues,
                'domain': domain,
                'input_metadata': input_metadata,
                'idea_response': item
            }
        return None
    except Exception as e:
        st.error(f"Error fetching idea capture data: {str(e)}")
        return None
```

Update the `get_lens_selector_data` function (around line 468) to include more data:

```python
def get_lens_selector_data(ai_request_id: str) -> Dict:
    """Get lens selector data from DynamoDB"""
    try:
        dynamodb, _, _ = init_aws_clients()
        if not dynamodb:
            return None
        
        table = dynamodb.Table(IDEAS_TABLE)
        response = table.get_item(Key={'ai_request_id': ai_request_id})
        
        if 'Item' in response:
            item = response['Item']
            lens_selector = item.get('lens_selector', {})
            results = lens_selector.get('results', [])
            input_metadata = item.get('input_metadata', {})
            domain = extract_domain(input_metadata)
            
            # Extract burning issues summary
            burning_issues = item.get('burning_issues', [])
            burning_issues_summary = [bi.get('title', '') for bi in burning_issues]
            
            return {
                'results': results,
                'domain': domain,
                'burning_issues_summary': burning_issues_summary,
                'input_data': input_metadata
            }
        return None
    except Exception as e:
        st.error(f"Error fetching lens selector data: {str(e)}")
        return None
```

Update the `get_survey_generator_data` function (around line 500) to include more data:

```python
def get_survey_generator_data(ai_request_id: str) -> Dict:
    """Get survey generator data from DynamoDB"""
    try:
        dynamodb, _, _ = init_aws_clients()
        if not dynamodb:
            return None
        
        table = dynamodb.Table(IDEAS_TABLE)
        response = table.get_item(Key={'ai_request_id': ai_request_id})
        
        if 'Item' in response:
            item = response['Item']
            survey_generator = item.get('survey_generator', {})
            questions = survey_generator.get('questions', [])
            input_data = survey_generator.get('input', {})
            domain = extract_domain(input_data)
            burning_issues = item.get('burning_issues', [])
            
            return {
                'questions': questions,
                'domain': domain,
                'burning_issues': burning_issues,
                'survey_data': survey_generator
            }
        return None
    except Exception as e:
        st.error(f"Error fetching survey generator data: {str(e)}")
        return None
```

Update the `get_360_report_data` function (around line 528) to include more data:

```python
def get_360_report_data(ai_request_id: str) -> Dict:
    """Get 360 report data from DynamoDB"""
    try:
        dynamodb, _, _ = init_aws_clients()
        if not dynamodb:
            return None
        
        table = dynamodb.Table(FINALIZE_REPORT_TABLE)
        response = table.get_item(Key={'ai_request_id': ai_request_id})
        
        if 'Item' in response:
            item = response['Item']
            report_json = json.loads(item.get('report_json', '{}'))
            verdict = report_json.get('verdict', {}).get('decision', 'Unknown')
            composite_score = report_json.get('verdict', {}).get('compositeScore', 0)
            confidence = report_json.get('verdict', {}).get('confidence', '0')
            
            return {
                'verdict': verdict,
                'composite_score': composite_score,
                'confidence': confidence,
                'report_data': report_json
            }
        return None
    except Exception as e:
        st.error(f"Error fetching 360 report data: {str(e)}")
        return None
```

Update the `get_social_lens_data` function (around line 555) to include more data:

```python
def get_social_lens_data(ai_request_id: str) -> Dict:
    """Get social lens data from DynamoDB"""
    try:
        dynamodb, _, _ = init_aws_clients()
        if not dynamodb:
            return None
        
        table = dynamodb.Table(SOCIAL_LENS_TABLE)
        response = table.get_item(Key={'request_id': ai_request_id})
        
        if 'Item' in response:
            item = response['Item']
            social_analysis = json.loads(item.get('social_lens_analysis_json', '{}'))
            buzz_score = social_analysis.get('overall_buzz_score', 0)
            buzz_level = social_analysis.get('buzz_level', 'Unknown')
            sources = social_analysis.get('sources', [])
            
            return {
                'buzz_score': buzz_score,
                'buzz_level': buzz_level,
                'sources': sources,
                'social_data': social_analysis
            }
        return None
    except Exception as e:
        st.error(f"Error fetching social lens data: {str(e)}")
        return None
```

Update the `get_match_maker_data` function (around line 582) to include more data:

```python
def get_match_maker_data(ai_request_id: str, persona_type: str) -> Dict:
    """Get match maker data from S3"""
    try:
        _, s3, _ = init_aws_clients()
        if not s3:
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
            return None
        
        # List objects
        objects = s3.list_objects_v2(
            Bucket=MATCH_MAKING_BUCKET,
            Prefix=prefix
        )
        
        if 'Contents' not in objects:
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
            return None
        
        # Get the file
        response = s3.get_object(Bucket=MATCH_MAKING_BUCKET, Key=latest_file)
        match_data = json.loads(response['Body'].read().decode('utf-8'))
        
        matches = match_data.get('matches', [])
        if matches:
            return {
                'selected_match': matches[0],  # Assume first is selected
                'matches': matches
            }
        
        return None
    except Exception as e:
        st.error(f"Error fetching match maker data: {str(e)}")
        return None
```

### 3. Replace Prompt Building Functions

Replace the `build_idea_capture_prompt` function (around line 105) with:

```python
def build_improved_idea_capture_prompt(burning_issues: List[Dict], domain: str, input_metadata: Dict, idea_response: Dict) -> str:
    """Build the improved prompt for the Idea Capture agent"""
    stage = input_metadata.get('stage', 'unknown')
    founder_goal = input_metadata.get('founder_goal', '')
    audience = input_metadata.get('audience', [])
    problem_statements = input_metadata.get('problemStatements', [])
    
    # Analyze burning issues quality
    issues_quality = analyze_burning_issues_quality(burning_issues)
    
    # Extract rationale and follow-up answers
    rationale = idea_response.get('finalize', {}).get('rationale', [])
    follow_up_answers = idea_response.get('input_metadata', {}).get('followUpAnswers', [])
    
    # Build the burning issues text with detailed analysis
    issues_text = ""
    for i, issue in enumerate(burning_issues, 1):
        title = issue.get('title', '')
        hypothesis = issue.get('hypothesis', '')
        why_it_matters = issue.get('why_it_matters', '')
        how_to_measure = issue.get('how_to_measure', '')
        success_threshold = issue.get('success_threshold', '')
        lens_signals = issue.get('lens_signals', [])
        
        issues_text += f"""
  Burning Issue #{i}: {title}
    Why it matters: {why_it_matters}
    Hypothesis: {hypothesis}
    How to measure: {how_to_measure}
    Success threshold: {success_threshold}
    Lens signals: {', '.join(lens_signals)}
    Quality assessment: {issues_quality.get(f'issue_{i}', 'Unknown')}
"""
    
    prompt = f"""Role:
You are an expert Meta-Feedback Analyst for Outlaw's Idea Capture agent. 
Your specialty is evaluating the quality, relevance, and actionability of AI-generated business insights.

Context Analysis:
AGENT: Idea Capture
DOMAIN: {domain}
STAGE: {stage}
FOUNDER GOAL: {founder_goal}
TARGET AUDIENCE: {', '.join(audience) if audience else 'Not specified'}

KEY PROBLEM STATEMENTS:
{chr(10).join(f"- {ps}" for ps in problem_statements[:3])}

AI OUTPUT ANALYSIS:
{issues_text}

RATIONALE PROVIDED:
{chr(10).join(f"- {r}" for r in rationale[:3])}

FOLLOW-UP ANSWERS QUALITY:
{analyze_follow_up_quality(follow_up_answers)}

Your Task:
Generate 2-3 incisive evaluation questions that assess the QUALITY and ACTIONABILITY of the AI's analysis.
For each question, create 4 specific, context-aware options that reflect different levels of performance quality.

Focus on:
1. Issue identification accuracy and depth
2. Hypothesis testability and relevance
3. Measurement approach practicality
4. Overall business insight quality
5. Alignment with founder's specific context

Question Requirements:
- Maximum 150 characters each
- Must evaluate AI performance quality
- Should uncover specific improvement areas
- Must be answerable by the founder
- Should provide actionable feedback for the AI team

Option Requirements:
- Create 4 distinct options per question
- Options should be specific to the context (domain, stage, burning issues)
- Avoid generic labels like "Good" or "Poor"
- Each option should clearly indicate a different level of performance
- Options should be actionable and provide clear feedback direction
- AI should generate both the value and label for each option

Format:
{{
  "questions": [
    {{
      "id": "q1",
      "text": "Specific evaluation question (max 150 chars)",
      "type": "multiple_choice",
      "options": [
        {{"value": "ai_generated_value_1", "label": "Context-specific excellent option"}},
        {{"value": "ai_generated_value_2", "label": "Context-specific good option"}},
        {{"value": "ai_generated_value_3", "label": "Context-specific fair option"}},
        {{"value": "ai_generated_value_4", "label": "Context-specific poor option"}}
      ],
      "rl_value": 0.85,
      "focus_area": "issue_identification"
    }}
  ],
  "analysis_summary": "Brief summary of what makes this output high/low quality"
}}

Generate evaluation questions now:"""
    
    return prompt
```

Replace the `build_lens_selector_prompt` function (around line 176) with:

```python
def build_improved_lens_selector_prompt(results: List[Dict], domain: str, burning_issues_summary: List[str], input_data: Dict) -> str:
    """Build the improved prompt for the Lens Selector agent"""
    stage = input_data.get('stage', 'unknown')
    tags = input_data.get('tags', [])
    
    # Analyze lens selection quality
    lens_quality = analyze_lens_selection_quality(results, burning_issues_summary)
    
    # Build detailed lens analysis
    lens_text = ""
    for i, lens in enumerate(results[:4], 1):
        lens_name = lens.get('lens', '')
        rank = lens.get('rank', 0)
        reason = lens.get('reason', '')
        confidence = lens.get('confidence', 0)
        pros = lens.get('pros', [])
        cons = lens.get('cons', [])
        stage_relevance = lens.get('stageRelevance', 0)
        
        lens_text += f"""
  Lens #{i}: {lens_name} (Rank: {rank})
    Reasoning: {reason}
    Confidence: {confidence:.1f}
    Stage relevance: {stage_relevance:.1f}
    Pros: {', '.join(pros[:2])}
    Cons: {', '.join(cons[:2])}
    Quality assessment: {lens_quality.get(f'lens_{i}', 'Unknown')}
"""
    
    prompt = f"""Role:
You are a Validation Strategy Expert evaluating Outlaw's Lens Selector agent performance.
Your expertise is in assessing the appropriateness of research methodologies for specific business contexts.

Context Analysis:
AGENT: Lens Selector
DOMAIN: {domain}
STAGE: {stage}
BUSINESS TAGS: {', '.join(tags)}
BURNING ISSUES FOCUS: {', '.join(burning_issues_summary[:2])}

LENS SELECTION ANALYSIS:
{lens_text}

LENS SELECTION QUALITY ASSESSMENT:
- Ranking logic coherence: {lens_quality.get('ranking_coherence', 'Unknown')}
- Stage appropriateness: {lens_quality.get('stage_appropriateness', 'Unknown')}
- Confidence justification: {lens_quality.get('confidence_justification', 'Unknown')}

Your Task:
Generate 2-3 evaluation questions that assess the STRATEGIC THINKING and METHODOLOGICAL APPROPRIATENESS of the lens selection.
For each question, create 4 specific, context-aware options that reflect different levels of strategic thinking quality.

Focus on:
1. Ranking logic and prioritization rationale
2. Stage-appropriateness of selected lenses
3. Confidence level justification
4. Alignment with burning issues
5. Overall research strategy quality

Question Requirements:
- Maximum 150 characters each
- Must evaluate strategic thinking quality
- Should assess methodological soundness
- Must provide actionable improvement insights

Option Requirements:
- Create 4 distinct options per question
- Options should be specific to the validation strategy context
- Avoid generic labels like "Good" or "Poor"
- Each option should clearly indicate a different level of strategic thinking
- Options should be actionable and provide clear feedback direction
- AI should generate both the value and label for each option

Format:
{{
  "questions": [
    {{
      "id": "q1",
      "text": "Strategic evaluation question (max 150 chars)",
      "type": "multiple_choice",
      "options": [
        {{"value": "ai_generated_value_1", "label": "Context-specific strategic excellence option"}},
        {{"value": "ai_generated_value_2", "label": "Context-specific strategic good option"}},
        {{"value": "ai_generated_value_3", "label": "Context-specific strategic fair option"}},
        {{"value": "ai_generated_value_4", "label": "Context-specific strategic poor option"}}
      ],
      "rl_value": 0.80,
      "focus_area": "strategic_thinking"
    }}
  ],
  "analysis_summary": "Assessment of lens selection strategic quality"
}}

Generate evaluation questions now:"""
    
    return prompt
```

Replace the `build_survey_generator_prompt` function (around line 237) with:

```python
def build_improved_survey_generator_prompt(questions: List[Dict], domain: str, burning_issues: List[Dict], survey_data: Dict) -> str:
    """Build the improved prompt for the Survey Generator agent"""
    stage = survey_data.get('input', {}).get('stage', 'unknown')
    survey_purpose = survey_data.get('input', {}).get('surveyPurpose', '')
    filters = survey_data.get('input', {}).get('filters', {})
    
    # Analyze survey quality
    survey_quality = analyze_survey_quality(questions, burning_issues)
    
    # Build detailed question analysis
    question_analysis = ""
    for i, q in enumerate(questions[:5], 1):
        q_text = q.get('text', '')
        q_type = q.get('type', '')
        bucket = q.get('bucket', '')
        bp_ref = q.get('burning_problem_reference', '')
        
        question_analysis += f"""
  Question #{i}: {q_text[:80]}...
    Type: {q_type}
    Bucket: {bucket}
    BP reference: {bp_ref}
    Quality: {survey_quality.get(f'q_{i}', 'Unknown')}
"""
    
    prompt = f"""Role:
You are a Survey Methodology Expert evaluating Outlaw's Survey Generator agent.
Your expertise is in assessing question design, survey flow, and research instrument quality.

Context Analysis:
AGENT: Survey Generator
DOMAIN: {domain}
STAGE: {stage}
SURVEY PURPOSE: {survey_purpose}
TARGET DEMOGRAPHIC: {json.dumps(filters, indent=2)}

BURNING ISSUES TARGETED:
{chr(10).join(f"- {bi.get('title', '')}" for bi in burning_issues[:3])}

SURVEY QUALITY ANALYSIS:
{question_analysis}

SURVEY DESIGN ASSESSMENT:
- Question clarity: {survey_quality.get('question_clarity', 'Unknown')}
- Burning issue alignment: {survey_quality.get('bi_alignment', 'Unknown')}
- Question type variety: {survey_quality.get('type_variety', 'Unknown')}
- Survey flow logic: {survey_quality.get('flow_logic', 'Unknown')}
- Demographic targeting: {survey_quality.get('demographic_targeting', 'Unknown')}

Your Task:
Generate 2-3 evaluation questions that assess the SURVEY DESIGN QUALITY and RESEARCH EFFECTIVENESS.
For each question, create 4 specific, context-aware options that reflect different levels of survey design quality.

Focus on:
1. Question clarity and unbiased design
2. Alignment with burning issues
3. Survey flow and respondent experience
4. Demographic targeting appropriateness
5. Overall research instrument quality

Question Requirements:
- Maximum 150 characters each
- Must evaluate survey design quality
- Should assess research effectiveness
- Must provide actionable improvement insights

Option Requirements:
- Create 4 distinct options per question
- Options should be specific to the survey design context
- Avoid generic labels like "Good" or "Poor"
- Each option should clearly indicate a different level of survey quality
- Options should be actionable and provide clear feedback direction
- AI should generate both the value and label for each option

Format:
{{
  "questions": [
    {{
      "id": "q1",
      "text": "Survey design evaluation question (max 150 chars)",
      "type": "multiple_choice",
      "options": [
        {{"value": "ai_generated_value_1", "label": "Context-specific survey excellence option"}},
        {{"value": "ai_generated_value_2", "label": "Context-specific survey good option"}},
        {{"value": "ai_generated_value_3", "label": "Context-specific survey fair option"}},
        {{"value": "ai_generated_value_4", "label": "Context-specific survey poor option"}}
      ],
      "rl_value": 0.85,
      "focus_area": "survey_design"
    }}
  ],
  "analysis_summary": "Assessment of survey design and research quality"
}}

Generate evaluation questions now:"""
    
    return prompt
```

Replace the `build_360_report_prompt` function (around line 294) with:

```python
def build_improved_360_report_prompt(verdict: str, composite_score: int, confidence: str, report_data: Dict) -> str:
    """Build the improved prompt for the 360 Report agent"""
    kpis = report_data.get('kpis', {})
    rationale = report_data.get('rationale', {})
    top_risks = report_data.get('topRisks', [])
    next_actions = report_data.get('topNextActions', [])
    
    # Analyze report quality
    report_quality = analyze_report_quality(report_data)
    
    prompt = f"""Role:
You are a Business Intelligence Expert evaluating Outlaw's 360 Report agent.
Your expertise is in assessing the quality, actionability, and strategic value of business recommendations.

Context Analysis:
AGENT: 360 Report
VERDICT: {verdict}
COMPOSITE SCORE: {composite_score}
CONFIDENCE LEVEL: {confidence}

KPIS BREAKDOWN:
- Time to insight: {kpis.get('timeToInsightMinutes', 'Unknown')} minutes
- Cost to learn: ${kpis.get('costToLearnUSD', 'Unknown')}
- Lens completion: {json.dumps(kpis.get('lens', []), indent=2)}

RATIONALE ANALYSIS:
- Overall score breakdown: {json.dumps(rationale.get('overall_score_breakdown', []), indent=2)}
- Risk penalty: {rationale.get('risk_penalty', 'Unknown')}
- Final confidence: {rationale.get('final_confidence', 'Unknown')}

KEY RISKS IDENTIFIED:
{chr(10).join(f"- {risk.get('severity', '')}: {risk.get('text', '')}" for risk in top_risks[:3])}

RECOMMENDED ACTIONS:
{chr(10).join(f"- {action}" for action in next_actions[:3])}

REPORT QUALITY ASSESSMENT:
- Verdict justification: {report_quality.get('verdict_justification', 'Unknown')}
- Risk identification: {report_quality.get('risk_identification', 'Unknown')}
- Action quality: {report_quality.get('action_quality', 'Unknown')}
- Overall coherence: {report_quality.get('overall_coherence', 'Unknown')}

Your Task:
Generate 2-3 evaluation questions that assess the STRATEGIC VALUE and ACTIONABILITY of the 360 report.
For each question, create 4 specific, context-aware options that reflect different levels of strategic value.

Focus on:
1. Verdict accuracy and justification
2. Risk identification completeness
3. Action item quality and feasibility
4. Overall strategic value
5. Report coherence and usefulness

Question Requirements:
- Maximum 150 characters each
- Must evaluate strategic value
- Should assess actionability
- Must provide actionable improvement insights

Option Requirements:
- Create 4 distinct options per question
- Options should be specific to the business context and verdict
- Avoid generic labels like "Good" or "Poor"
- Each option should clearly indicate a different level of strategic value
- Options should be actionable and provide clear feedback direction
- AI should generate both the value and label for each option

Format:
{{
  "questions": [
    {{
      "id": "q1",
      "text": "Strategic value evaluation question (max 150 chars)",
      "type": "multiple_choice",
      "options": [
        {{"value": "ai_generated_value_1", "label": "Context-specific strategic excellence option"}},
        {{"value": "ai_generated_value_2", "label": "Context-specific strategic good option"}},
        {{"value": "ai_generated_value_3", "label": "Context-specific strategic fair option"}},
        {{"value": "ai_generated_value_4", "label": "Context-specific strategic poor option"}}
      ],
      "rl_value": 0.98,
      "focus_area": "strategic_value"
    }}
  ],
  "analysis_summary": "Assessment of 360 report strategic value"
}}

Generate evaluation questions now:"""
    
    return prompt
```

Replace the `build_social_lens_prompt` function (around line 342) with:

```python
def build_improved_social_lens_prompt(buzz_score: int, buzz_level: str, sources: List[str], social_data: Dict) -> str:
    """Build the improved prompt for the Social Lens agent"""
    main_talking_points = social_data.get('main_talking_points', [])
    topics_getting_hotter = social_data.get('topics_getting_hotter', [])
    group_sentiment = social_data.get('group_sentiment', [])
    worries = social_data.get('worries', [])
    
    # Analyze social analysis quality
    social_quality = analyze_social_quality(social_data)
    
    prompt = f"""Role:
You are a Social Intelligence Expert evaluating Outlaw's Social Synth agent.
Your expertise is in assessing the quality, relevance, and insightfulness of social media and trend analysis.

Context Analysis:
AGENT: Social Synth
BUZZ SCORE: {buzz_score}
BUZZ LEVEL: {buzz_level}
SOURCES ANALYZED: {len(sources)}
SOURCE TYPES: {', '.join(set(sources))}

KEY TALKING POINTS:
{chr(10).join(f"- {point.get('title', '')} ({point.get('mood', '')} mood)" for point in main_talking_points[:3])}

TRENDING TOPICS:
{chr(10).join(f"- {topic.get('topic', '')} ({topic.get('change_percent', '')}% change)" for topic in topics_getting_hotter[:3])}

SENTIMENT BREAKDOWN:
{chr(10).join(f"- {group.get('group', '')}: {group.get('positive_percent', '')}% positive" for group in group_sentiment[:3])}

KEY WORRIES IDENTIFIED:
{chr(10).join(f"- {worry.get('issue', '')} ({worry.get('severity', '')} severity)" for worry in worries[:3])}

SOCIAL ANALYSIS QUALITY ASSESSMENT:
- Buzz scoring accuracy: {social_quality.get('buzz_scoring', 'Unknown')}
- Source relevance: {social_quality.get('source_relevance', 'Unknown')}
- Sentiment analysis: {social_quality.get('sentiment_analysis', 'Unknown')}
- Trend identification: {social_quality.get('trend_identification', 'Unknown')}
- Overall insight quality: {social_quality.get('overall_insight', 'Unknown')}

Your Task:
Generate 2-3 evaluation questions that assess the INSIGHT QUALITY and RELEVANCE of the social analysis.
For each question, create 4 specific, context-aware options that reflect different levels of insight quality.

Focus on:
1. Buzz score accuracy and relevance
2. Source selection and analysis quality
3. Sentiment analysis accuracy
4. Trend identification value
5. Overall business insight usefulness

Question Requirements:
- Maximum 150 characters each
- Must evaluate insight quality
- Should assess business relevance
- Must provide actionable improvement insights

Option Requirements:
- Create 4 distinct options per question
- Options should be specific to the social analysis context
- Avoid generic labels like "Good" or "Poor"
- Each option should clearly indicate a different level of insight quality
- Options should be actionable and provide clear feedback direction
- AI should generate both the value and label for each option

Format:
{{
  "questions": [
    {{
      "id": "q1",
      "text": "Social insight evaluation question (max 150 chars)",
      "type": "multiple_choice",
      "options": [
        {{"value": "ai_generated_value_1", "label": "Context-specific insight excellence option"}},
        {{"value": "ai_generated_value_2", "label": "Context-specific insight good option"}},
        {{"value": "ai_generated_value_3", "label": "Context-specific insight fair option"}},
        {{"value": "ai_generated_value_4", "label": "Context-specific insight poor option"}}
      ],
      "rl_value": 0.90,
      "focus_area": "social_insight"
    }}
  ],
  "analysis_summary": "Assessment of social analysis insight quality"
}}

Generate evaluation questions now:"""
    
    return prompt
```

Replace the `build_match_maker_prompt` function (around line 391) with:

```python
def build_improved_match_maker_prompt(selected_match: Dict, match_score: float, all_matches: List[Dict], persona_type: str) -> str:
    """Build the improved prompt for the Match-Maker agent"""
    match_name = selected_match.get('name', 'Unknown')
    match_expertise = selected_match.get('expertise', [])
    match_background = selected_match.get('background', '')
    
    # Analyze matching quality
    match_quality = analyze_matching_quality(selected_match, all_matches, persona_type)
    
    # Build comparison with other top matches
    comparison = ""
    for i, match in enumerate(all_matches[:3], 1):
        m_name = match.get('name', 'Unknown')
        m_score = match.get('match_score', 0)
        m_expertise = match.get('expertise', [])[:2]
        comparison += f"""
  Match #{i}: {m_name}
    Score: {m_score:.1f}
    Expertise: {', '.join(m_expertise)}
    Selected: {'Yes' if match == selected_match else 'No'}
"""
    
    prompt = f"""Role:
You are an Expert Matching Specialist evaluating Outlaw's Match-Maker agent.
Your expertise is in assessing the quality, relevance, and appropriateness of expert recommendations.

Context Analysis:
AGENT: Match-Maker
PERSONA TYPE: {persona_type}
SELECTED MATCH: {match_name}
MATCH SCORE: {match_score}
SELECTED EXPERTISE: {', '.join(match_expertise)}
BACKGROUND: {match_background}

TOP MATCHES COMPARISON:
{comparison}

MATCHING QUALITY ASSESSMENT:
- Score justification: {match_quality.get('score_justification', 'Unknown')}
- Expertise relevance: {match_quality.get('expertise_relevance', 'Unknown')}
- Persona fit: {match_quality.get('persona_fit', 'Unknown')}
- Overall match quality: {match_quality.get('overall_quality', 'Unknown')}
- Alternative consideration: {match_quality.get('alternative_consideration', 'Unknown')}

Your Task:
Generate 2-3 evaluation questions that assess the MATCHING QUALITY and RELEVANCE of the expert recommendation.
For each question, create 4 specific, context-aware options that reflect different levels of matching quality.

Focus on:
1. Match score accuracy and justification
2. Expertise relevance to needs
3. Persona type appropriateness
4. Overall recommendation quality
5. Value compared to alternatives

Question Requirements:
- Maximum 150 characters each
- Must evaluate matching quality
- Should assess relevance to needs
- Must provide actionable improvement insights

Option Requirements:
- Create 4 distinct options per question
- Options should be specific to the expert matching context
- Avoid generic labels like "Good" or "Poor"
- Each option should clearly indicate a different level of matching quality
- Options should be actionable and provide clear feedback direction
- AI should generate both the value and label for each option

Format:
{{
  "questions": [
    {{
      "id": "q1",
      "text": "Match quality evaluation question (max 150 chars)",
      "type": "multiple_choice",
      "options": [
        {{"value": "ai_generated_value_1", "label": "Context-specific matching excellence option"}},
        {{"value": "ai_generated_value_2", "label": "Context-specific matching good option"}},
        {{"value": "ai_generated_value_3", "label": "Context-specific matching fair option"}},
        {{"value": "ai_generated_value_4", "label": "Context-specific matching poor option"}}
      ],
      "rl_value": 0.92,
      "focus_area": "matching_quality"
    }}
  ],
  "analysis_summary": "Assessment of expert matching quality"
}}

Generate evaluation questions now:"""
    
    return prompt
```

### 4. Update the generate_feedback_questions Function

Replace the `generate_feedback_questions` function (around line 645) with:

```python
def generate_feedback_questions(stage: str, ai_request_id: str, persona_type: str = None) -> List[Dict]:
    """Generate feedback questions based on stage"""
    if stage == 'idea_capture':
        data = get_idea_capture_data(ai_request_id)
        if data:
            prompt = build_improved_idea_capture_prompt(
                data['burning_issues'], 
                data['domain'], 
                data['input_metadata'],
                data['idea_response']
            )
            response = invoke_bedrock(prompt)
            if response:
                try:
                    questions_data = json.loads(response)
                    return questions_data.get('questions', [])
                except json.JSONDecodeError:
                    st.error("Failed to parse Bedrock response")
    
    elif stage == 'lens_selector':
        data = get_lens_selector_data(ai_request_id)
        if data:
            prompt = build_improved_lens_selector_prompt(
                data['results'], 
                data['domain'], 
                data['burning_issues_summary'],
                data['input_data']
            )
            response = invoke_bedrock(prompt)
            if response:
                try:
                    questions_data = json.loads(response)
                    return questions_data.get('questions', [])
                except json.JSONDecodeError:
                    st.error("Failed to parse Bedrock response")
    
    elif stage == 'survey_generator':
        data = get_survey_generator_data(ai_request_id)
        if data:
            prompt = build_improved_survey_generator_prompt(
                data['questions'], 
                data['domain'], 
                data['burning_issues'],
                data['survey_data']
            )
            response = invoke_bedrock(prompt)
            if response:
                try:
                    questions_data = json.loads(response)
                    return questions_data.get('questions', [])
                except json.JSONDecodeError:
                    st.error("Failed to parse Bedrock response")
    
    elif stage == '360_report':
        data = get_360_report_data(ai_request_id)
        if data:
            prompt = build_improved_360_report_prompt(
                data['verdict'], 
                data['composite_score'], 
                data['confidence'],
                data['report_data']
            )
            response = invoke_bedrock(prompt)
            if response:
                try:
                    questions_data = json.loads(response)
                    return questions_data.get('questions', [])
                except json.JSONDecodeError:
                    st.error("Failed to parse Bedrock response")
    
    elif stage == 'social_lens':
        data = get_social_lens_data(ai_request_id)
        if data:
            prompt = build_improved_social_lens_prompt(
                data['buzz_score'], 
                data['buzz_level'], 
                data['sources'],
                data['social_data']
            )
            response = invoke_bedrock(prompt)
            if response:
                try:
                    questions_data = json.loads(response)
                    return questions_data.get('questions', [])
                except json.JSONDecodeError:
                    st.error("Failed to parse Bedrock response")
    
    elif stage == 'match_maker':
        if persona_type:
            data = get_match_maker_data(ai_request_id, persona_type)
            if data:
                prompt = build_improved_match_maker_prompt(
                    data['selected_match'], 
                    data['selected_match'].get('match_score', 0),
                    data['matches'],
                    persona_type
                )
                response = invoke_bedrock(prompt)
                if response:
                    try:
                        questions_data = json.loads(response)
                        return questions_data.get('questions', [])
                    except json.JSONDecodeError:
                        st.error("Failed to parse Bedrock response")
    
    return []
```

## Summary of Changes

1. **Added quality analysis functions** for each agent stage to evaluate output quality before generating questions.

2. **Enhanced data fetching** to include more context and detailed information from agent outputs.

3. **Replaced all prompt building functions** with improved versions that:
   - Provide more context to the AI
   - Generate dynamic option values instead of using hardcoded ones
   - Create context-specific options that are relevant to each agent stage
   - Include quality assessment in the prompt generation process

4. **Updated the main question generation function** to use the improved prompt building functions.

These changes will significantly improve the quality of the generated feedback questions by making them more context-specific and relevant to each agent stage's output.