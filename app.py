import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

# 1. TẢI VÀ CHUẨN BỊ DỮ LIỆU
@st.cache_data
def load_data():
    ratings = pd.read_csv('ratings.csv')
    movies = pd.read_csv('movies.csv')
    
    train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)
    user_item_matrix = train_data.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    
    knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
    knn_model.fit(user_item_matrix.values)
    
    # Lấy danh sách tất cả các thể loại (loại bỏ giá trị rỗng/không hợp lệ)
    all_genres = set()
    for genres in movies['genres'].dropna().str.split('|'):
        all_genres.update(genres)
    all_genres.discard('(no genres listed)')
    sorted_genres = sorted(list(all_genres))
    
    # Tính toán trước bảng thống kê (Popularity) cho toàn bộ phim để dùng chung
    stats = train_data.groupby('movieId').agg(
        rating_count=('rating', 'count'),
        rating_mean=('rating', 'mean')
    )
    
    return ratings, movies, train_data, user_item_matrix, knn_model, sorted_genres, stats

ratings_df, movies_df, train_df, matrix, knn, genre_list, movie_stats = load_data()

def get_movie_titles(movie_ids, movies_df):
    return movies_df[movies_df['movieId'].isin(movie_ids)][['movieId', 'title', 'genres']]



# 2. CÁC PHƯƠNG PHÁP GỢI Ý

# Phương pháp 1: Gợi ý các phim đang phổ biến
def get_popular_movies(stats, movies_df, min_ratings=50, top_n=10):
    popular = stats[stats['rating_count'] >= min_ratings].sort_values(by='rating_mean', ascending=False).head(top_n)
    return get_movie_titles(popular.index.tolist(), movies_df)

# Phương pháp 2: Gợi ý cá nhân hóa (K-NN Collaborative Filtering)
def get_cf_recommendations(user_id, matrix, model, movies_df, top_n=10):
    if user_id not in matrix.index:
        return None 
    
    user_index = matrix.index.get_loc(user_id)
    distances, indices = model.kneighbors(matrix.iloc[user_index, :].values.reshape(1, -1), n_neighbors=6)
    
    similar_users_indices = indices.flatten()[1:]
    similar_users_movies = matrix.iloc[similar_users_indices]
    
    movie_scores = similar_users_movies.mean(axis=0)
    user_seen_movies = matrix.iloc[user_index]
    movie_scores = movie_scores[user_seen_movies == 0]
    
    top_movie_ids = movie_scores.nlargest(top_n).index.tolist()
    return get_movie_titles(top_movie_ids, movies_df)

# Phương pháp 3: Gợi ý theo thể loại cho người dùng mới
def get_genre_recommendations(selected_genres, movies_df, stats, top_n=10):
    # Lọc phim có chứa ít nhất một trong các thể loại được chọn
    pattern = '|'.join(selected_genres)
    filtered_movies = movies_df[movies_df['genres'].str.contains(pattern, case=False, na=False)]
    
    # Kết hợp với bảng thống kê điểm số
    merged = pd.merge(filtered_movies, stats, on='movieId')
    
    # Lọc các phim có ít nhất 20 lượt đánh giá để đảm bảo chất lượng, sau đó sắp xếp theo điểm trung bình
    top_movies = merged[merged['rating_count'] >= 20].sort_values(by='rating_mean', ascending=False).head(top_n)
    return top_movies[['movieId', 'title', 'genres']]



# 3. GIAO DIỆN STREAMLIT
st.title("🎬 Movie Recommender Mini")
st.write("Hệ thống gợi ý phim cá nhân hóa (KNN) và bài toán Cold-Start.")
st.write("Nhóm 11 vippro(Hải vip nhất)")
st.sidebar.header("Tùy chọn User")
user_type = st.sidebar.radio("Phân loại người dùng:", ["Người dùng đã có dữ liệu", "Người dùng mới"])

if user_type == "Người dùng đã có dữ liệu":
    user_list = sorted(matrix.index.tolist())
    selected_user = st.sidebar.selectbox("Chọn User ID:", user_list)
else:
    selected_genres = st.sidebar.multiselect("Chọn thể loại yêu thích của bạn:", genre_list, default=["Action", "Sci-Fi"])

if st.button("Lấy danh sách gợi ý"):
    st.markdown("---")
    
    if user_type == "Người dùng đã có dữ liệu":
        st.subheader(f"Gợi ý cá nhân hóa cho User ID: {selected_user}")
        cf_recs = get_cf_recommendations(selected_user, matrix, knn, movies_df, top_n=10)
        
        if cf_recs is not None and not cf_recs.empty:
            st.markdown(cf_recs.set_index('movieId').to_html(), unsafe_allow_html=True)
        else:
            st.info("Chưa đủ dữ liệu.")
            
    else:
        st.subheader(f"Gợi ý theo thể loại: {', '.join(selected_genres)}")
        if selected_genres:
            genre_recs = get_genre_recommendations(selected_genres, movies_df, movie_stats, top_n=10)
            if not genre_recs.empty:
                st.markdown(genre_recs.set_index('movieId').to_html(), unsafe_allow_html=True)
            else:
                st.warning("Không tìm thấy phim phù hợp với thể loại này hoặc chưa đủ lượt đánh giá.")
        else:
            st.warning("Vui lòng chọn ít nhất một thể loại từ menu bên trái.")
            
    st.markdown("---")
    st.subheader("Gợi ý phim thịnh hành chung (Popularity Baseline)")
    pop_recs = get_popular_movies(movie_stats, movies_df, top_n=10)
    st.markdown(pop_recs.set_index('movieId').to_html(), unsafe_allow_html=True)