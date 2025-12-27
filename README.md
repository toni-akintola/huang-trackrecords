# Huang Track Records Model

A bandit model created by [Alice C.W. Huang](https://alicecwhuang.github.io/), which depicts how scientific networks whose agents depend heavily upon track records to make judgments about the credence of hypotheses may be less accurate than networks whose agents rely more on evidential reasoning to judge hypotheses. View the full paper [here](https://philarchive.org/archive/HUATRA-2).

## Abstract

> In the literature on expert trust, it is often assumed that track records are the gold
> standard for evaluating expertise, and the difficulty of expert identification arises from
> either the lack of access to track records, or the inability to assess them (Goldman [2001];
> Schurz [2012]; Nguyen [2020]). I show, using a computational model, that even in an
> idealized environment where agents have a God’s eye view on track records, they may fail
> to identify experts. Under plausible conditions, selecting testimony based on track records
> ends up reducing overall accuracy, and preventing the community from identifying the
> real experts.

## Model Details

### Mechanism

The model simulates scientists estimating the bias of a coin (a proportion they are trying to determine). Each agent:

- Updates beliefs based on coin tosses they observe (evidence)
- Solicits testimony from other agents based on track records (social learning)
- Combines evidential and social components using parameter `c`: `P(H) = (1-c) * P_social(H) + c * P_evidential(H)`

The hypothesis space consists of 6 mutually exclusive hypotheses about the coin bias: 0, 0.2, 0.4, 0.6, 0.8, and 1.0. At each time step, agents:

1. Observe a coin toss (evidence)
2. Update their evidential component using Bayesian updating with noise
3. Select informants based on track records (variation-dependent)
4. Update their social component by averaging informants' credences
5. Combine evidential and social components
6. Update their track record based on prediction accuracy

### Key Concepts

- **Track Records**: Publicly available history of prediction accuracy, measured using Brier scores
- **Meta-expertise**: The community's ability to identify accurate agents, measured by the R-value (correlation between centrality/authority and accuracy)
- **Authority Scores**: Computed using the HITS algorithm on the trust network (who solicits testimony from whom)

### Model Variations

1. **Track Record Scientist** (`tr-scientist`):
   - Agents select informants based on track records (choosing the top m% performers)
   - Agents update beliefs based on both evidence and testimony (c varies between 0 and 1)
   - This is the main "testimony model" from the paper, showing how track-record-based testimony selection can reduce accuracy and meta-expertise due to premature convergence

2. **Random Scientist** (`random-scientist`):
   - Agents randomly select informants rather than choosing based on track records
   - This represents the "less-monopoly" modification from the paper, reducing the influence of early lucky agents
   - Maintains opinion diversity at the cost of sometimes consulting less accurate peers

3. **Patient Scientist** (`patient-scientist`):
   - Agents work in complete isolation (c = 1 for all agents)
   - They assess others based on track records but do NOT update their beliefs based on testimony
   - This represents the "baseline model" from the paper, serving as a control to isolate the effects of testimony exchange

### Parameters

- `Num Nodes`: Number of scientists in the network (default: 30)
- `Graph Type`: Network structure - "complete", "cycle", or "wheel" (default: "complete")
- `Truth`: True coin bias (one of 0, 0.2, 0.4, 0.6, 0.8, or 1.0)
- `Feedback Rate`: Probability that track record is updated after each prediction (default: 1.0)
- `Model Variation`: Which variation to run - "tr-scientist", "random-scientist", or "patient-scientist"

### Agent Attributes

Each scientist maintains:

- `Credence`: Credence distribution over the 6 hypotheses (sums to 1)
- `Brier Score`: Current Brier score measuring accuracy of beliefs
- `Brier History`: History of prediction Brier scores
- `Record`: Track record (updated predictions when feedback_rate > 0)
- `R Value`: Community-wide correlation between centrality and accuracy (meta-expertise metric)
- `M`: Open-mindedness parameter (0.05 to 0.5) - percentage of community willing to consult
- `C`: Weight on evidence vs. testimony (0 to 1) - 1 = isolation, 0 = pure testimony
- `Noise`: Noise parameter (0.001 to 0.2) - deviation from ideal Bayesian updating


**Click the 'Visualizations' tab to get started.**