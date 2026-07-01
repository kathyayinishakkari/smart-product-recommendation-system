from rapidfuzz import process
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
class RecommendationEngine:
    """
    Content-Based Product Recommendation Engine
    """

    def __init__(self):
        self.df = None
        self.tfidf = TfidfVectorizer(
            stop_words="english",
            max_features=5000
        )
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.indices = None
        self.max_rating = None
        self.max_rating_count = None

    def load_data(self, file_path):
        """
        Load the cleaned dataset.
        """

        self.df = pd.read_csv(file_path)
        self.max_rating = self.df["rating"].max()
        self.max_rating_count = self.df["rating_count"].max()
        print(f"Dataset loaded successfully!")

        print(f"Shape: {self.df.shape}")

    def create_tfidf_matrix(self):
        """
        Create TF-IDF feature matrix.
        """

        self.tfidf_matrix = self.tfidf.fit_transform(
        self.df["combined_features"]
        )

        print("TF-IDF Matrix Created")

        print(self.tfidf_matrix.shape)

    def compute_similarity(self):
        """
        Compute cosine similarity matrix.
        """

        self.cosine_sim = cosine_similarity(
            self.tfidf_matrix
        )

        print("Cosine Similarity Matrix Created")

        print(self.cosine_sim.shape)

    def create_indices(self):
        """
        Create product index.
        """

        self.indices = pd.Series(
            self.df.index,
            index=self.df["product_name"]
        )

        print("Indices Created")

    def search_product(self, query: str):
        """
        Search for a product using keyword search first,
        then fall back to fuzzy matching.
        """

        query = query.lower().strip()

        # ---------- Keyword Search ----------
        matches = self.df[
            self.df["combined_features"]
            .str.lower()
            .str.contains(query, na=False)
        ]

        if not matches.empty:
            return matches.iloc[0]["product_name"]

        # ---------- Fuzzy Search ----------
        product_names = self.df["product_name"].tolist()

        result = process.extractOne(
            query,
            product_names,
            score_cutoff=60
        )

        if result:
            return result[0]

        return None
    
    def generate_explanation(self, similarity, product):
        """
        Generate a human-readable explanation for why a product
        was recommended.
        """

        reasons = []

        if similarity >= 0.70:
            reasons.append("Very similar to your search")
        elif similarity >= 0.50:
            reasons.append("Similar product")
        else:
            reasons.append("Related product")

        if product["rating"] >= 4.5:
            reasons.append("Highly rated")

        elif product["rating"] >= 4.0:
            reasons.append("Well rated")

        if product["rating_count"] >= 10000:
            reasons.append("Popular among customers")

        elif product["rating_count"] >= 1000:
            reasons.append("Frequently reviewed")

        return " | ".join(reasons)
    
    def calculate_recommendation_score(self, final_score, best_final_score):
        """
        Calculate recommendation score using the final weighted score.
        """

        score = (final_score / best_final_score) * 100

        score = min(score, 100)

        if score >= 95:
            badge = "🥇 Top Pick"

        elif score >= 90:
            badge = "⭐ Excellent"

        elif score >= 80:
            badge = "👍 Highly Recommended"

        elif score >= 70:
            badge = "✅ Good Choice"

        else:
            badge = "🔍 Worth Considering"

        return round(score, 1), badge
    
    def recommend(
    self,
    query,
    top_n=10,
    min_rating=0,
    max_price=float("inf"),
    min_discount=0,
    category=None
):
        """
        Main recommendation pipeline.
        """

        similarity_scores = self.get_similarity_scores(
            query,
            top_n
        )

        if similarity_scores is None:
            return pd.DataFrame()

        scored_products = self.calculate_final_scores(
            similarity_scores
        )

        recommendations = self.build_recommendations(
            scored_products
        )

        recommendations = self.apply_filters(
            recommendations,
            min_rating,
            max_price,
            min_discount,
            category
        )

        print(
            f"Found {len(recommendations)} matching recommendations."
        )

        if recommendations.empty:
            print(
                "No products matched your filters."
            )

        return recommendations
    
    def train(self, file_path):
        """
        Complete training pipeline.
        """

        self.load_data(file_path)

        self.create_tfidf_matrix()

        self.compute_similarity()

        self.create_indices()

        print("Recommendation Engine Ready!")
    
    def save_model(self, model_dir="../models"):
        """
        Save all trained model objects.
        """

        joblib.dump(self.tfidf, f"{model_dir}/tfidf_vectorizer.pkl")

        joblib.dump(self.tfidf_matrix, f"{model_dir}/tfidf_matrix.pkl")

        joblib.dump(self.cosine_sim, f"{model_dir}/cosine_similarity.pkl")

        joblib.dump(self.indices, f"{model_dir}/product_indices.pkl")

        joblib.dump(self.df, f"{model_dir}/products_dataframe.pkl")

        print("Model saved successfully!")

    def load_model(self, model_dir="../models"):
        """
        Load all trained model objects.
        """

        self.tfidf = joblib.load(f"{model_dir}/tfidf_vectorizer.pkl")

        self.tfidf_matrix = joblib.load(f"{model_dir}/tfidf_matrix.pkl")

        self.cosine_sim = joblib.load(f"{model_dir}/cosine_similarity.pkl")

        self.indices = joblib.load(f"{model_dir}/product_indices.pkl")

        self.df = joblib.load(f"{model_dir}/products_dataframe.pkl")

        self.max_rating = self.df["rating"].max()

        self.max_rating_count = self.df["rating_count"].max()

        print("Model loaded successfully!")

    def get_similarity_scores(self, query, top_n):
        """
        Find similar products based on cosine similarity.
        """

        product_name = self.search_product(query)

        if product_name is None:
            return None

        idx = self.indices[product_name]

        similarity_scores = list(
            enumerate(self.cosine_sim[idx])
        )

        similarity_scores = sorted(
            similarity_scores,
            key=lambda x: x[1],
            reverse=True
        )

        return similarity_scores[1:top_n + 1]
    
    def calculate_final_scores(self, similarity_scores):
        """
        Calculate weighted scores for every recommendation.
        """

        scored_products = []

        for product_index, similarity in similarity_scores:

            product = self.df.iloc[product_index]

            rating_score = (
                product["rating"] /
                self.max_rating
            )

            popularity_score = (
                product["rating_count"] /
                self.max_rating_count
            )

            final_score = (
                0.70 * similarity +
                0.20 * rating_score +
                0.10 * popularity_score
            )

            scored_products.append(
                (
                    product_index,
                    similarity,
                    final_score
                )
            )

        return scored_products
    
    def build_recommendations(self, scored_products):
        """
        Convert scored products into a dataframe.
        """

        best_final_score = max(
            score[2]
            for score in scored_products
        )

        recommendations = []

        for product_index, similarity, final_score in scored_products:

            product = self.df.iloc[product_index]

            explanation = self.generate_explanation(
                similarity,
                product
            )

            recommendation_score, badge = (
                self.calculate_recommendation_score(
                    final_score,
                    best_final_score
                )
            )

            recommendations.append({

                "Product": product["product_name"],

                "Category": product["category"],

                "Price": product["discounted_price"],

                "Actual Price": product["actual_price"],

                "Discount (%)": product["discount_percentage"],

                "Rating": product["rating"],

                "Rating Count": product["rating_count"],

                "Similarity": round(float(similarity),4),

                "Final Score": round(float(final_score),4),

                "Recommendation Score": recommendation_score,

                "Badge": badge,

                "Why Recommended": explanation,

                "Image": product["img_link"],

                "Amazon Link": product["product_link"]

            })

        recommendations = pd.DataFrame(recommendations)

        recommendations = recommendations.sort_values(
            by="Final Score",
            ascending=False
        ).reset_index(drop=True)

        return recommendations
    
    def apply_filters(
    self,
    recommendations,
    min_rating,
    max_price,
    min_discount,
    category
):
        """
        Apply user-selected filters.
        """

        recommendations = recommendations[
            recommendations["Rating"] >= min_rating
        ]

        recommendations = recommendations[
            recommendations["Price"] <= max_price
        ]

        recommendations = recommendations[
            recommendations["Discount (%)"] >= min_discount
        ]

        if category is not None:

            recommendations = recommendations[
                recommendations["Category"]
                .str.contains(
                    category,
                    case=False,
                    na=False
                )
            ]

        return recommendations.reset_index(drop=True)