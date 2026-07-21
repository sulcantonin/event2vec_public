"""Life-path event-sequence dataset (faithful to the paper's life-event graph).

The transition graph and domain (life-stage) labels are transcribed from the
paper's authoritative Graphviz source ``paper/i/main.dot`` so the synthetic
Life Path data is reproducible from the installed package alone. Each entry of
``LIFEPATH_TRANSITIONS`` maps an event to a list of ``(next_event, weight)``
pairs (weights are the transition probabilities shown on the graph edges).
``LIFEPATH_DOMAINS`` maps each event to its life-stage cluster, used as a
ground-truth label for downstream probes (experiment E1) and hierarchy
analysis (E3).
"""
from typing import Dict, List, Tuple

import torch

from ..data import generate_sequences

INITIAL = "birth"
TERMINAL = "death"

LIFEPATH_TRANSITIONS: Dict[str, List[Tuple[str, float]]] = {
    'academic_achievement': [('graduation', 0.6), ('scholarship', 0.2), ('internship', 0.2)],
    'adoption': [('child_milestones', 0.6), ('financial_hardship', 0.2), ('relationship_challenge', 0.2)],
    'apprenticeship': [('career_start', 0.7), ('job_search', 0.3)],
    'birth': [('infancy', 1.0)],
    'business_failure': [('job_search', 0.5), ('debt', 0.3), ('career_change', 0.2)],
    'business_success': [('financial_success', 0.5), ('investment', 0.3), ('leadership_role', 0.2)],
    'career_change': [('job_search', 0.5), ('vocational_training', 0.3), ('entrepreneurship', 0.2)],
    'career_start': [('promotion', 0.3), ('relationship', 0.3), ('job_change', 0.3), ('health_issue', 0.1)],
    'child_milestones': [('parenthood', 0.5), ('financial_hardship', 0.3), ('relationship_challenge', 0.2)],
    'college': [('graduation', 0.6), ('study_abroad', 0.1), ('internship', 0.2), ('health_issue', 0.1)],
    'college_application': [('college', 0.7), ('scholarship', 0.2), ('job_search', 0.1)],
    'creative_pursuit': [('hobby_development', 0.5), ('friendship', 0.3), ('career_change', 0.2)],
    'dating': [('relationship', 0.6), ('friendship', 0.3), ('travel', 0.1)],
    'debt': [('financial_hardship', 0.6), ('job_change', 0.3), ('relationship_challenge', 0.1)],
    'disability': [('financial_hardship', 0.4), ('health_issue', 0.3), ('recovery', 0.2), ('retirement', 0.1)],
    'divorce': [('dating', 0.4), ('financial_hardship', 0.3), ('relocation', 0.2), ('health_issue', 0.1)],
    'early_childhood': [('middle_childhood', 0.8), ('health_issue', 0.1), ('elementary_school', 0.1)],
    'elder_years': [('health_issue', 0.4), ('grandparenthood', 0.3), ('legacy_planning', 0.2), ('terminal_illness', 0.1)],
    'elementary_school': [('middle_school', 0.8), ('friendship', 0.1), ('health_issue', 0.1)],
    'engagement': [('marriage', 0.8), ('relationship_challenge', 0.2)],
    'entrepreneurship': [('business_success', 0.4), ('business_failure', 0.3), ('financial_hardship', 0.2), ('leadership_role', 0.1)],
    'financial_hardship': [('debt', 0.4), ('job_search', 0.3), ('health_issue', 0.2), ('relationship_challenge', 0.1)],
    'financial_success': [('investment', 0.4), ('major_purchase', 0.3), ('travel', 0.2), ('retirement', 0.1)],
    'first_job': [('career_start', 0.6), ('job_change', 0.2), ('relationship', 0.2)],
    'friendship': [('dating', 0.5), ('travel', 0.3), ('hobby_development', 0.2)],
    'graduate_school': [('graduation', 0.7), ('academic_achievement', 0.2), ('health_issue', 0.1)],
    'graduation': [('job_search', 0.4), ('first_job', 0.3), ('graduate_school', 0.2), ('travel', 0.1)],
    'grandparenthood': [('elder_years', 0.5), ('legacy_planning', 0.3), ('health_issue', 0.2)],
    'health_issue': [('recovery', 0.5), ('disability', 0.2), ('financial_hardship', 0.2), ('terminal_illness', 0.1)],
    'high_school': [('college_application', 0.5), ('graduation', 0.2), ('vocational_training', 0.2), ('health_issue', 0.1)],
    'hobby_development': [('friendship', 0.4), ('creative_pursuit', 0.3), ('travel', 0.2), ('relationship', 0.1)],
    'hospice': [('death', 1.0)],
    'infancy': [('toddlerhood', 0.9), ('health_issue', 0.1)],
    'inheritance': [('financial_success', 0.5), ('investment', 0.3), ('major_purchase', 0.2)],
    'internship': [('job_search', 0.5), ('first_job', 0.3), ('college', 0.2)],
    'investment': [('financial_success', 0.5), ('major_purchase', 0.3), ('retirement', 0.2)],
    'job_change': [('relocation', 0.3), ('promotion', 0.3), ('relationship', 0.2), ('financial_success', 0.2)],
    'job_loss': [('job_search', 0.6), ('financial_hardship', 0.3), ('health_issue', 0.1)],
    'job_search': [('first_job', 0.3), ('job_change', 0.3), ('financial_hardship', 0.2), ('relocation', 0.2)],
    'late_childhood': [('middle_school', 0.8), ('health_issue', 0.1), ('friendship', 0.1)],
    'leadership_role': [('business_success', 0.3), ('financial_success', 0.3), ('major_purchase', 0.2), ('health_issue', 0.2)],
    'legacy_planning': [('elder_years', 0.4), ('terminal_illness', 0.3), ('health_issue', 0.3)],
    'major_purchase': [('financial_hardship', 0.4), ('debt', 0.3), ('financial_success', 0.3)],
    'marriage': [('parenthood', 0.4), ('relocation', 0.2), ('relationship_challenge', 0.2), ('major_purchase', 0.2)],
    'middle_childhood': [('late_childhood', 0.7), ('elementary_school', 0.2), ('health_issue', 0.1)],
    'middle_school': [('high_school', 0.8), ('friendship', 0.1), ('health_issue', 0.1)],
    'military_service': [('career_start', 0.4), ('relationship', 0.2), ('health_issue', 0.1)],
    'parenthood': [('child_milestones', 0.4), ('financial_hardship', 0.2), ('relationship_challenge', 0.2), ('job_change', 0.2)],
    'promotion': [('leadership_role', 0.4), ('relocation', 0.3), ('relationship', 0.2), ('health_issue', 0.1)],
    'recovery': [('job_change', 0.3), ('relationship', 0.3), ('spiritual_journey', 0.2), ('hobby_development', 0.2)],
    'relationship': [('engagement', 0.4), ('dating', 0.3), ('relocation', 0.2), ('relationship_challenge', 0.1)],
    'relationship_challenge': [('divorce', 0.4), ('marriage', 0.3), ('health_issue', 0.2)],
    'relocation': [('job_change', 0.3), ('relationship', 0.3), ('friendship', 0.2), ('travel', 0.2)],
    'retirement': [('travel', 0.3), ('hobby_development', 0.3), ('elder_years', 0.2), ('grandparenthood', 0.2)],
    'scholarship': [('college', 0.7), ('academic_achievement', 0.3)],
    'spiritual_journey': [('volunteer_work', 0.4), ('relationship', 0.3), ('travel', 0.2), ('creative_pursuit', 0.1)],
    'study_abroad': [('college', 0.6), ('travel', 0.3), ('relationship', 0.1)],
    'terminal_illness': [('hospice', 0.7), ('death', 0.3)],
    'toddlerhood': [('early_childhood', 0.9), ('health_issue', 0.1)],
    'travel': [('relationship', 0.3), ('friendship', 0.3), ('creative_pursuit', 0.2), ('spiritual_journey', 0.2)],
    'vocational_training': [('graduation', 0.7), ('apprenticeship', 0.2), ('health_issue', 0.1)],
    'volunteer_work': [('friendship', 0.4), ('spiritual_journey', 0.3), ('career_change', 0.2), ('relationship', 0.1)],
    'widowhood': [('health_issue', 0.4), ('financial_hardship', 0.3), ('relocation', 0.2), ('dating', 0.1)],
}

LIFEPATH_DOMAINS: Dict[str, str] = {
    'academic_achievement': 'Education',
    'adoption': 'Relationships',
    'apprenticeship': 'Career',
    'birth': 'EarlyLife',
    'business_failure': 'Career',
    'business_success': 'Career',
    'career_change': 'Career',
    'career_start': 'Career',
    'child_milestones': 'Relationships',
    'college': 'Education',
    'college_application': 'Education',
    'creative_pursuit': 'LifeEvents',
    'dating': 'Relationships',
    'death': 'Terminal',
    'debt': 'HealthFinances',
    'disability': 'HealthFinances',
    'divorce': 'Relationships',
    'early_childhood': 'EarlyLife',
    'elder_years': 'LaterLife',
    'elementary_school': 'Education',
    'engagement': 'Relationships',
    'entrepreneurship': 'Career',
    'financial_hardship': 'HealthFinances',
    'financial_success': 'HealthFinances',
    'first_job': 'Career',
    'friendship': 'Relationships',
    'graduate_school': 'Education',
    'graduation': 'Education',
    'grandparenthood': 'LaterLife',
    'health_issue': 'HealthFinances',
    'high_school': 'Education',
    'hobby_development': 'LifeEvents',
    'hospice': 'LaterLife',
    'infancy': 'EarlyLife',
    'inheritance': 'HealthFinances',
    'internship': 'Career',
    'investment': 'HealthFinances',
    'job_change': 'Career',
    'job_loss': 'Career',
    'job_search': 'Career',
    'late_childhood': 'EarlyLife',
    'leadership_role': 'Career',
    'legacy_planning': 'LaterLife',
    'major_purchase': 'HealthFinances',
    'marriage': 'Relationships',
    'middle_childhood': 'EarlyLife',
    'middle_school': 'Education',
    'military_service': 'LifeEvents',
    'parenthood': 'Relationships',
    'promotion': 'Career',
    'recovery': 'HealthFinances',
    'relationship': 'Relationships',
    'relationship_challenge': 'Relationships',
    'relocation': 'LifeEvents',
    'retirement': 'LaterLife',
    'scholarship': 'Education',
    'spiritual_journey': 'LifeEvents',
    'study_abroad': 'Education',
    'terminal_illness': 'LaterLife',
    'toddlerhood': 'EarlyLife',
    'travel': 'LifeEvents',
    'vocational_training': 'Education',
    'volunteer_work': 'LifeEvents',
    'widowhood': 'Relationships',
}

# Canonical event vocabulary (sorted for a stable index mapping across runs).
EVENT_TYPES: List[str] = sorted(
    set(LIFEPATH_TRANSITIONS)
    | {d for outs in LIFEPATH_TRANSITIONS.values() for d, _ in outs}
)


def make_lifepath(num_seq: int = 10000, max_length: int = 16, seed: int = 0):
    """Generate the Life Path dataset as the uniform dataset record.

    Returns a dict ``{sequences, processed_sequences, vocab, inv_vocab,
    labels, graph}`` where ``labels`` are integer domain ids per event id and
    ``graph`` is ``LIFEPATH_TRANSITIONS`` (the ground-truth transition graph).
    """
    import random as _random

    state = _random.getstate()
    _random.seed(seed)
    try:
        raw = generate_sequences(
            EVENT_TYPES, LIFEPATH_TRANSITIONS, INITIAL, TERMINAL, num_seq, max_length
        )
    finally:
        _random.setstate(state)

    vocab = {e: i for i, e in enumerate(EVENT_TYPES)}
    inv_vocab = {i: e for e, i in vocab.items()}
    processed = [
        torch.tensor([vocab[e] for e in seq], dtype=torch.long) for seq in raw
    ]
    domain_names = sorted(set(LIFEPATH_DOMAINS.values()))
    domain_id = {d: i for i, d in enumerate(domain_names)}
    labels = {
        vocab[e]: domain_id[LIFEPATH_DOMAINS[e]]
        for e in EVENT_TYPES
        if e in LIFEPATH_DOMAINS
    }
    return {
        "sequences": raw,
        "processed_sequences": processed,
        "vocab": vocab,
        "inv_vocab": inv_vocab,
        "labels": labels,
        "domain_names": domain_names,
        "graph": LIFEPATH_TRANSITIONS,
    }
