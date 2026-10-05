# Prior Art and Defensible Novelty Boundary

This document maps EQ-Layer against pre-LLM dialogue-system research. Its goal
is not to manufacture a "first" claim. It records which ideas clearly existed
before modern large language models, which combinations are close to EQ-Layer,
and where a narrower contribution may still be defensible.

## Bottom line

EQ-Layer is **not** the first system to:

- infer a user's goal or dialogue act;
- maintain dialogue state across turns;
- detect affect, frustration, uncertainty, or satisfaction;
- adapt dialogue strategy to a user model;
- select a dialogue action before surface realization;
- use confidence/uncertainty in action selection;
- repair misunderstandings or react to corrections;
- adapt content, politeness, or social behavior to the user;
- optimize a dialogue policy;
- evaluate alternative dialogue strategies with human users.

Those ideas are established pre-LLM research topics.

The potentially defensible contribution is narrower:

> A model-agnostic control layer for frozen/general-purpose generative language
> models that explicitly factor intent, affect, conversational subtext, stance
> evidence, and uncertainty into inspectable state; select a discrete
> interaction policy before token generation; steer the unchanged base model;
> preserve provenance for direct/derived/structural/verified signals; and test
> the layer using same-base-model blinded response-level comparisons.

This wording is intentionally a **novelty hypothesis**, not a first-in-history
claim. It must still be checked against contemporary LLM middleware and
steering literature.

---

## High-value pre-LLM prior art

### 1. TRAINS / grounding / repair (1994-1996)

**Key work**
- Traum & Allen (1994), formal theory of repair.
- Allen et al. (1995), TRAINS conversational planning agent.
- Traum & Dillenbourg (1996), miscommunication and repair.
- Traum & Heeman (1996), grounding in spoken dialogue.

**What already existed**
- explicit discourse context;
- shared plans and discourse obligations;
- mixed-initiative action selection;
- models of grounding, misunderstanding, and repair;
- repair as a consequence of misaligned agent mental states.

**EQ-Layer overlap**
- correction/repair policies;
- conversational history;
- stance/grounding-related state;
- separating a dialogue move from its surface wording.

**What to borrow**
- represent repair as a **state transition**, not merely a keyword class;
- add explicit `misalignment_probability`, `repair_target`, and
  `repair_distance` fields;
- distinguish self-correction from user-corrects-system repair.

**Reference**
- https://trips.ihmc.us/trips/TRAINS.html
- David Traum publication archive: https://people.ict.usc.edu/~traum/Papers/

### 2. How May I Help You? / open-ended intent routing (1997)

**Key work**
- Gorin, Riccardi, Wright et al., *How May I Help You?*, Speech Communication,
  1997.

**What already existed**
- open-ended user utterances;
- automatic semantic action / call-type classification;
- learning from large sets of real user interactions rather than requiring
  users to speak in a fixed grammar.

**EQ-Layer overlap**
- learned intent classification;
- routing a free-form user request into a compact control state.

**What to borrow**
- evaluate intent on naturally occurring user language;
- replace synthetic intent data with a genuinely held-out human corpus where
  possible.

**DOI**
- 10.1016/S0167-6393(97)00040-X

### 3. PARADISE dialogue evaluation (1997-1998)

**Key work**
- Walker, Litman, Kamm & Abella (1997), PARADISE.

**What already existed**
- systematic comparison of dialogue strategies;
- separation of task requirements from dialogue behavior;
- user satisfaction as an empirical outcome;
- combining task success and interaction costs.

**EQ-Layer overlap**
- blinded response-level evaluation;
- explicit separation of component scores from final user-facing performance.

**What to borrow**
- add task success / constraint satisfaction as an objective endpoint alongside
  human preference;
- report interaction cost: turns, tokens, clarification count, and latency;
- do not reduce evaluation to "empathetic preference" alone.

**DOI**
- 10.3115/976909.979652

### 4. COLLAGEN / collaborative user-goal structure (1998)

**Key work**
- Rich & Sidner (1998), COLLAGEN.

**What already existed**
- application-independent collaboration management;
- hierarchical interaction history organized around user and agent goals;
- mixed-initiative assistance.

**EQ-Layer overlap**
- model-agnostic control layer;
- intent and action organization outside the surface generator.

**What to borrow**
- represent user goals hierarchically instead of a single flat
  `canonical_request`;
- allow subgoals and abandoned/corrected goals to persist explicitly.

**Reference**
- MERL TR97-21a, *COLLAGEN: A Collaboration Manager for Software Interface
  Agents*.

### 5. GoDiS / information-state dialogue management (1999-2000)

**Key work**
- Larsson et al. (2000), *GoDiS - An Accommodating Dialogue System*.

**What already existed**
- explicit information state;
- update phase separated from move selection;
- private/shared state;
- questions under discussion;
- user information could arrive in flexible order.

**EQ-Layer overlap**
- state -> policy -> realization pipeline;
- explicit intermediate state instead of direct generation.

**What to borrow**
- split state into **private control state** and **shared commitments**;
- add a Questions Under Discussion (QUD)-like stack for unresolved user goals;
- make clarification target a named unresolved slot/reference.

**Reference**
- ACL Anthology W00-0302.

### 6. Corrections in spoken dialogue systems / TOOT (2000-2002)

**Key work**
- Swerts, Litman & Hirschberg (2000), *Corrections in Spoken Dialogue Systems*.
- Litman & Pan (2002), *Designing and Evaluating an Adaptive Spoken Dialogue
  System*.

**What already existed**
- empirical study of user corrections of system errors;
- evidence that correction behavior depends on dialogue strategy;
- dynamically changing user models;
- automatic dialogue-strategy adaptation when recognition problems occur.

**EQ-Layer overlap**
- structural correction handling;
- repair policy;
- dynamic, turn-by-turn state.

**What to borrow**
- classify correction type and its target;
- track repeated system failure separately from user affect;
- use error/recovery history as an independent policy feature.

**DOIs**
- Corrections: 10.21437/ICSLP.2000-344
- Adaptive TOOT: 10.1023/A:1015036910358

### 7. Emotional cues inside dialogue management (2002)

**Key work**
- Holzapfel, Fuegen, Denecke & Waibel (2002),
  *Integrating Emotional Cues into a Framework for Dialogue Management*.

**What already existed**
- emotion represented explicitly inside dialogue management;
- emotional cues used as information for dialogue behavior;
- design intended to transfer across domains/tasks.

**EQ-Layer overlap**
- affect as an explicit intermediate control signal;
- affect influencing policy rather than merely changing wording.

**Novelty implication**
- EQ-Layer must **not** claim to be the first system that inserts affect into a
  dialogue-management layer.

**DOI**
- 10.1109/ICMI.2002.1166983

### 8. RavenClaw (2003; extended 2009)

**Key work**
- Bohus & Rudnicky (2003/2009), RavenClaw.

**What already existed**
- reusable, task-independent dialogue engine;
- explicit separation of task logic from domain-independent conversational
  behavior;
- error handling, turn taking, and conversational skills outside domain logic.

**EQ-Layer overlap**
- reusable middleware;
- discrete policy before realization;
- task semantics separated from conversational handling.

**What to borrow**
- keep domain/task action separate from social/conversational action;
- treat the control layer as a reusable engine, not a bag of prompts.

**DOIs**
- 10.21437/Eurospeech.2003-255
- 10.1016/j.csl.2008.10.001

### 9. User-tailored response generation / MATCH-SPUR (2004)

**Key work**
- Walker et al. (2004), *Generation and Evaluation of User Tailored Responses
  in Multimodal Dialogue*.

**What already existed**
- explicit user models;
- response content adapted to the user;
- content selection and conciseness manipulated algorithmically;
- experimental evidence that tailored responses can improve efficacy.

**EQ-Layer overlap**
- user constraints/preferences;
- verbosity/directness control;
- response realization conditioned on a user model.

**What to borrow**
- make content selection and surface style **two separate decisions**;
- evaluate concise/direct versus verbose/supportive adaptations independently.

**DOI**
- 10.1207/s15516709cog2805_8

### 10. Relational agents and embodied social dialogue (2000-2006)

**Key work**
- Cassell et al., *Embodied Conversational Agents* (2000).
- Bickmore & Cassell (2000), social dialogue with embodied agents.
- Bickmore dissertation (2003), relational agents.
- Bickmore & Picard (2005), long-term human-computer relationships.

**What already existed**
- social dialogue separated from task dialogue;
- relationship-building behavior;
- adaptation over repeated interactions;
- verbal and non-verbal social signals.

**EQ-Layer overlap**
- social calibration;
- non-patronizing / relationship-aware response behavior;
- conversational policy beyond pure task completion.

**What to borrow**
- optional session-level relational state;
- distinguish immediate affect from longer-term interaction history;
- do not collapse rapport, politeness, empathy, and affect into one scalar "EQ".

### 11. POMDP dialogue management / uncertainty (2007)

**Key work**
- Williams & Young (2007), *Partially Observable Markov Decision Processes for
  Spoken Dialog Systems*.

**What already existed**
- multiple uncertain dialogue-state hypotheses;
- confidence scores;
- policy/action selection under uncertainty;
- globally optimized behavior rather than hard deterministic state.

**EQ-Layer overlap**
- confidence-gated intent;
- ambiguity handling;
- policy selection.

**What to borrow**
- replace single-label confidence with a **belief distribution** over plausible
  intent/subtext states;
- propagate uncertainty into policy choice;
- clarification should be an action selected because expected loss is lower,
  not only because a fixed threshold fired.

**DOI**
- 10.1016/j.csl.2006.06.008

### 12. Affect-sensitive AutoTutor (2006-2011)

**Key work**
- D'Mello et al. (2006-2008), affect detection in AutoTutor.
- D'Mello, Picard & Graesser (2007), affect-sensitive AutoTutor.
- later real-time uncertainty detection and adaptive responses.

**What already existed**
- simultaneous cognitive + affective user state;
- detection of boredom, confusion, frustration, engagement/flow;
- confidence in affect classification;
- previous affect state;
- selection of pedagogical/motivational dialogue moves based on those states.

**This is one of the closest pre-LLM architectural relatives to EQ-Layer.**

**EQ-Layer overlap**
- intent/cognitive state + affective state;
- history;
- confidence;
- state-conditioned discrete dialogue move.

**What to borrow**
- explicit `previous_affect` and persistence duration;
- separate cognitive/task state from affect state all the way through policy
  selection;
- evaluate whether adaptation helps only when affect detection is correct.

**References**
- MIT Media Lab, *AutoTutor Detects and Responds to Learners Affective and
  Cognitive States* (2008).
- Speech Communication (2011), real-time uncertainty detection/adaptation.

### 13. Affective dialogue management with factored POMDPs (2009)

**Key work**
- Bui, Zwiers, Poel & Nijholt (2009), *Affective Dialogue Management Using
  Factored POMDPs*.

**What already existed**
- user affect explicitly factored into dialogue state;
- affect changes system action selection;
- proof-of-concept policy optimization.

**EQ-Layer overlap**
- affect + task state -> policy.

**Novelty implication**
- any broad claim such as "first architecture to choose dialogue policy from
  intent and emotion" is indefensible.

**What to borrow**
- factor state/action spaces instead of using a single monolithic policy label.

### 14. Quality-adaptive dialogue (2014)

**Key work**
- Ultes, Dikme & Minker, quality-adaptive dialogue.

**What already existed**
- online estimate of interaction quality/user satisfaction;
- adaptation of confirmation strategy based on current interaction quality;
- focus on rescuing problematic dialogue trajectories.

**EQ-Layer overlap**
- escalation/quality trend;
- dynamic policy adaptation.

**What to borrow**
- add a rolling `interaction_quality` state independent of emotion;
- distinguish "user is angry" from "interaction is failing";
- use recovery latency and quality improvement as evaluation endpoints.

**Reference**
- ACL Anthology L14-1092.

### 15. NEMO / externally coupled affect adaptation (pre-modern-LLM)

**Key line of work**
- need-inspired task-independent emotion model (NEMO) integrated externally
  with a spoken conversational agent.

**Why it matters**
- affective adaptation did not always require rewriting the dialogue manager;
  external modules could alter the host system.

**EQ-Layer overlap**
- middleware/wrapper architecture.

**Novelty implication**
- "external affect module attached to an existing conversational system" is not
  by itself new.

### 16. Sentiment-adaptive end-to-end dialogue (2018)

**Key work**
- Shi & Yu (2018), *Sentiment Adaptive End-to-End Dialog Systems*.

**What already existed**
- multimodal user sentiment included directly in dialogue learning;
- both supervised and reinforcement-learning variants;
- sentiment adaptation improved task success and reduced dialogue length in the
  reported bus-information setting.

**EQ-Layer overlap**
- learned user affect modifying dialogue behavior.

**What to borrow**
- this is a strong pre-modern-LLM baseline family for any claim that affect
  adaptation improves task-oriented dialogue.

**Reference**
- ACL Anthology P18-1140.

---

## Architectural comparison

| Capability | Clear pre-LLM precedent? | Representative prior art | EQ-Layer status |
| --- | --- | --- | --- |
| Free-form intent routing | Yes | How May I Help You? | Learned intent |
| Dialogue-act classification | Yes | classic SDS / GoDiS | Learned |
| Explicit dialogue state | Yes | TRAINS, GoDiS | Yes |
| Uncertainty/confidence | Yes | POMDP dialogue systems | Partial |
| User model | Yes | TOOT, MATCH, relational agents | Partial |
| Affect detection | Yes | affective computing, AutoTutor | Yes |
| Affect changes policy | Yes | Holzapfel, AutoTutor, Bui | Yes |
| Correction/repair | Yes | TRAINS/grounding, TOOT | Structural repair lifecycle implemented |
| Task vs discourse separation | Yes | RavenClaw | Partial |
| Social/relationship behavior | Yes | REA, relational agents | Partial |
| User-tailored content/style | Yes | MATCH/SPUR | Partial |
| Satisfaction/quality adaptation | Yes | PARADISE, Ultes et al. | Structural interaction-quality state implemented |
| Learned policy optimization | Yes | NJFun, POMDP work | Not current core |
| External affect middleware | Yes | NEMO line | Yes-ish |
| Frozen general-purpose LLM steering | No pre-LLM analogue by definition | — | Yes |
| Same base model A/B isolation | not an architectural novelty | evaluation methodology | Implemented |
| Evidence provenance: direct/derived/structural/verified | no clear direct match found in this review | — | Implemented |
| Explicit no-go preservation in routing evidence | no clear direct match found in this review | — | Implemented |

---

## What EQ-Layer should change next

These are the highest-value lessons from the historical literature.

### A. Factored action architecture — IMPLEMENTED

EQ-Layer now compiles the selected policy into:

`state + intent -> task_move + social_move + repair_move + realization_controls`

The legacy policy selector remains for backward-compatible routing, but it no
longer carries the full response decision by itself. `task_move` is derived
primarily from explicit user intent, while social and repair behavior are
compiled separately. This prevents affective adaptation from silently replacing
the user's task.

Implemented in:
- `eq_layer/actions.py`
- `eval/factored_action_eval.py`

Example:

```text
task_move      = answer_status
social_move    = low_validation
repair_move    = none
verbosity      = low
directness     = high
clarification  = false
```

This carries forward the factorization lessons from RavenClaw and affective
POMDP work and explicitly prevents "emotion policy" from replacing the user's
task. Core CI enforces this orthogonality.

### B. Interaction Quality as a separate variable — IMPLEMENTED

Conversation failure is no longer inferred from emotion alone. EQ-Layer now
tracks a separate structural state:

```text
interaction_quality:
  current
  delta
  repeated_failure_count
  unresolved_repair_count
  clarification_count
  evidence_level
```

The initial implementation is intentionally conservative. It uses observable
repair pressure and clarification overhead rather than pretending to have a
learned satisfaction model. A low interaction-quality score can shorten and
make realization more direct, but it is explicitly not interpreted as user
emotion.

Implemented in:
- `eq_layer/interaction.py`
- `eval/interaction_state_eval.py`

This carries forward the distinction emphasized by quality-adaptive dialogue
work: "the user is emotionally activated" and "the interaction is failing" are
different variables.

### C. Structural repair lifecycle — IMPLEMENTED

Correction is now relational rather than only a flat subtext label:

```text
repair:
  active
  kind
  target_turn_index
  target_excerpt
  correction_excerpt
  repeated
  recent_repair_count
  confidence
  evidence_level
```

A correction becomes active only when an observable user correction follows a
prior assistant turn. Once the assistant responds, the repair is labelled
`responded_repair` rather than "resolved": the system does not claim that its
reply actually fixed the misunderstanding without later evidence. Repeated
repairs compile to `stop_restatement_and_repair`.

Crucially, repair state does **not** imply `user_right` or `user_wrong`;
factual stance remains externally verified or unknown.

Implemented in:
- `eq_layer/interaction.py`
- `eq_layer/tracker.py`
- `eq_layer/actions.py`
- `eval/interaction_state_eval.py`
- `eval/factored_action_eval.py`

### D. Move from confidence threshold to belief state

Instead of:

```text
intent = status_check
confidence = .61
```

prefer:

```text
intent_belief:
  status_check: .61
  question: .27
  explanation: .08
  other: .04
```

Policy selection can then compare expected loss of acting versus asking.

### E. Separate short-term affect from long-term user model

Do not store "the user is angry" as a stable user property.

Use:
- turn/session affect;
- session interaction quality;
- optional persistent user preferences only when appropriate.

This avoids conflating affective computing with relational/user modeling.

### F. Expand evaluation beyond preference

Add:
- intent/constraint satisfaction;
- task success;
- repair success;
- recovery latency;
- number of unnecessary clarifications;
- response length/token overhead;
- latency;
- user preference;
- interaction quality before/after the response.

This combines lessons from PARADISE, adaptive TOOT, and affect-sensitive
systems.

---

## Claims to avoid

Do **not** write:

- "the first emotionally intelligent dialogue manager";
- "the first system to use emotion to choose a response";
- "the first architecture combining intent and affect";
- "the first adaptive dialogue policy based on user emotion";
- "the first middleware that adapts dialogue to user state" without a much
  narrower qualifier;
- "EQ-Layer gives LLMs emotional intelligence" as an empirical conclusion.

All have strong prior-art problems or exceed current evidence.

## Claims that may be defensible after contemporary prior-art checking

Safer formulation:

> EQ-Layer investigates whether an explicit, model-agnostic control layer can
> improve frozen general-purpose LLM responses by separating user intent,
> affective state, conversational evidence, discrete interaction-policy
> selection, and generation steering.

Stronger but still testable formulation:

> Unlike prompt-only empathy instructions, EQ-Layer exposes the intermediate
> control state and policy decision, preserves the provenance and uncertainty of
> those signals, and can be evaluated by holding the underlying language model
> fixed across baseline and controlled conditions.

Do not add "first" unless a dedicated contemporary systematic search supports
it.

---

## Core references

1. Picard, R. W. (1997). *Affective Computing*. MIT Press.
2. Gorin et al. (1997). *How May I Help You?* Speech Communication.
3. Walker et al. (1997). *PARADISE: A Framework for Evaluating Spoken Dialogue
   Agents*. ACL/EACL.
4. Rich & Sidner (1998). *COLLAGEN: A Collaboration Manager for Software
   Interface Agents*. User Modeling and User-Adapted Interaction.
5. Larsson et al. (2000). *GoDiS - An Accommodating Dialogue System*.
6. Swerts, Litman & Hirschberg (2000). *Corrections in Spoken Dialogue Systems*.
7. Litman & Pan (2002). *Designing and Evaluating an Adaptive Spoken Dialogue
   System*.
8. Holzapfel et al. (2002). *Integrating Emotional Cues into a Framework for
   Dialogue Management*.
9. Bohus & Rudnicky (2003/2009). RavenClaw.
10. Walker et al. (2004). *Generation and Evaluation of User Tailored Responses
    in Multimodal Dialogue*.
11. Bickmore & Picard (2005). Long-term relational agents.
12. Williams & Young (2007). POMDPs for spoken dialogue systems.
13. D'Mello et al. (2007/2008). Affect-sensitive AutoTutor.
14. Gratch et al. (2007). *Creating Rapport with Virtual Agents*.
15. Bui et al. (2009). *Affective Dialogue Management Using Factored POMDPs*.
16. Ultes et al. (2014). Quality-adaptive dialogue.
17. Shi & Yu (2018). *Sentiment Adaptive End-to-End Dialog Systems*.

---

## Current research conclusion

The historical record strongly supports the following interpretation:

**EQ-Layer is an integration/reframing contribution, not a de-novo invention
of affective dialogue management.**

Its strongest research path is therefore:

1. make the integration technically cleaner than historical systems;
2. exploit properties unique to modern general-purpose LLMs;
3. keep the base model frozen so the layer's causal contribution can be
   isolated;
4. expose state/policy decisions for inspection;
5. preserve uncertainty and evidence provenance;
6. demonstrate response-level gains under blinded same-model evaluation.

That is a substantially more defensible research story than claiming that
intent-aware or emotion-aware dialogue policy itself is new.
