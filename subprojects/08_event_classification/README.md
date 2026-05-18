# SP08 — Event Classification

**Status:** Pending  
**Script:** `classify_events.py`  
**Outputs:** `../../results/event_classification/`

Classifies each detected geopolitical shock into one of four categories: Absorbed, Localized, Systemic, or Market-only stress. Combines shock intensity, EMFI response, and HMM stress probability in a [t, t+5] window around each shock.

See `ACTION_PLAN.md` Phase 8 for full specification.
