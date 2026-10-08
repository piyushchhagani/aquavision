# AquaVision Dataset Comparison

| Dataset | Water Surface | Detection | Segmentation | Diversity | Primary Candidate |
|---|---|---|---|---|---|
| TACO | Low/Medium | Yes | Yes | High | No |
| TrashCan | Low for surface / High for marine debris | Yes | Yes | High | No |
| Water-surface-specific dataset | Required | Required | Preferred | Required | To evaluate |
| AquaVision Local Dataset | High | To annotate | Future | High local relevance | Future test set |

## Current Decision

TACO and TrashCan are supplementary candidates.

The primary dataset must be specifically relevant to visible pollution on water surfaces.

## Selection Rule

Do not finalize the model classes until the primary dataset annotations have been inspected.
