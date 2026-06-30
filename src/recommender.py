from rapidfuzz import process
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
    
    def recommend(self,query,top_n=10,min_rating=0,max_price=float("inf"),min_discount=0,category=None):
        """
        Recommend similar products.
        """

        product_name = self.search_product(query)

        if product_name is None:
            return pd.DataFrame()

        idx = self.indices[product_name]

        similarity_scores = list(enumerate(self.cosine_sim[idx]))

        similarity_scores = sorted(
            similarity_scores,
            key=lambda x: x[1],
            reverse=True
        )

        similarity_scores = similarity_scores[1:top_n + 1]

        recommendations = []

        for product_index, similarity in similarity_scores:

            product = self.df.iloc[product_index]
            explanation = self.generate_explanation( similarity,product)
            rating_score = product["rating"] / self.max_rating

            popularity_score = (
                product["rating_count"] /
                self.max_rating_count
            )

            final_score = (
                0.70 * similarity +
                0.20 * rating_score +
                0.10 * popularity_score
            )

            recommendations.append({

                "Product": product["product_name"],

                "Category": product["category"],

                "Price": product["discounted_price"],

                "Actual Price": product["actual_price"],

                "Discount (%)": product["discount_percentage"],

                "Rating": product["rating"],

                "Rating Count": product["rating_count"],

                "Similarity": round(float(similarity), 4),

                "Final Score": round(float(final_score), 4),

                "Why Recommended": explanation,

                "Image": product["img_link"],

                "Amazon Link": product["product_link"]

            })

        recommendations = pd.DataFrame(recommendations)
        recommendations = recommendations.sort_values(
            by="Final Score",
            ascending=False
        ).reset_index(drop=True)
        # Filter by minimum rating
        recommendations = recommendations[
            recommendations["Rating"] >= min_rating
        ]

        # Filter by maximum price
        recommendations = recommendations[
            recommendations["Price"] <= max_price
        ]

        # Filter by minimum discount
        recommendations = recommendations[
            recommendations["Discount (%)"] >= min_discount
        ]

        # Filter by category
        if category is not None:
            recommendations = recommendations[
                recommendations["Category"]
                .str.contains(category, case=False, na=False)
            ]

        recommendations = recommendations.reset_index(drop=True)
        print(f"Found {len(recommendations)} matching recommendations.")
        if recommendations.empty:
            print("No products matched your filters.")
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