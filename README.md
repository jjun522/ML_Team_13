# Beer Recommendation System (Team 13)

## Project Overview
We train a beer recommender on the BeerAdvocate dataset so that each user receives beers that match their taste profile. The model is purely data‑driven: it learns from user–beer interactions, review scores, and basic metadata. The Brewer’s Friend recipe dataset is used later for validation and analysis (for example, checking whether recommended beers share similar ABV or IBU with what the user liked).

## Datasets
- **BeerAdvocate Reviews (`beer_reviews.csv`)** – ~1.5M reviews across ~66K beers with per‑aspect scores and metadata. Used to build the collaborative and hybrid models.
- **Brewer’s Friend Recipes (`recipeData.csv`)** – ~75K homebrew recipes with style, method, and analytic stats (ABV, IBU, OG, FG, Color, etc.). Used for content features, persona analysis, and cross‑checks.

## Modeling Strategy
1. **Model-based Collaborative Filtering**  
   Matrix Factorization (SVD from `surprise`) learns user and beer latent factors from the user–beer rating matrix. The model predicts ratings for unseen beers.
2. **Content-based Filtering**  
   Numerical recipe attributes (ABV, IBU, OG, FG, Color) plus one-hot style features are scaled and compared with cosine similarity to surface beers with similar intrinsic properties.
3. **Hybrid Filtering**  
   CF predictions and CBF similarities are combined with simple weights so we can rank beers that are both well matched to the user history and close in content space.

## Evaluation
- **Quantitative**: chronological 80/20 split of BeerAdvocate data. Metrics include RMSE, Precision@K, Recall@K, and NDCG@K (stored in `src/result/evaluation_metrics.json`).
- **Qualitative**: persona-style inspections (e.g., “IPA lover”) plus Brewer’s Friend cross-checks to confirm recommended beers share realistic brewing traits.

## Web Result Visualization
- `python -m http.server 8000` to check web page
