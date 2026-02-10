import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def preprocess_music_data():
    # 1. 데이터 로드
    try:
        music_tcc = pd.read_csv('tcc_ceds_music.csv')
        music_spot = pd.read_csv('spotify_tracks_analyzed.csv')
    except FileNotFoundError as e:
        print(f"파일을 찾을 수 없습니다: {e}")
        return

    # 2. 병합을 위한 키 생성 (가수명||곡명 소문자 처리)
    music_spot = music_spot.rename(columns={'artists': 'artist_name'})

    music_tcc['merge_key'] = music_tcc['artist_name'].str.lower().str.strip() + "||" + music_tcc[
        'track_name'].str.lower().str.strip()
    music_spot['merge_key'] = music_spot['artist_name'].str.lower().str.strip() + "||" + music_spot[
        'track_name'].str.lower().str.strip()

    # 3. 데이터 병합 (Outer Join)
    merged_music = pd.merge(music_tcc, music_spot, on='merge_key', how='outer', suffixes=('_tcc', '_spot'))

    # 4. 필수 컬럼 통합 및 결측치 처리
    merged_music['artist_name'] = merged_music['artist_name_spot'].fillna(merged_music['artist_name_tcc'])
    merged_music['track_name'] = merged_music['track_name_spot'].fillna(merged_music['track_name_tcc'])
    merged_music = merged_music.dropna(subset=['artist_name', 'track_name'])

    # 오디오 특성 결측치 보완 (Spotify 우선, 없으면 TCC)
    audio_features = ['danceability', 'energy', 'loudness', 'acousticness', 'instrumentalness', 'valence']
    for col in audio_features:
        col_tcc = f"{col}_tcc"
        col_spot = f"{col}_spot"
        if col_tcc in merged_music.columns and col_spot in merged_music.columns:
            merged_music[col] = merged_music[col_spot].fillna(merged_music[col_tcc])

    merged_music = merged_music.dropna(subset=audio_features)

    # 기타 컬럼 정리
    if 'track_genre' in merged_music.columns:
        merged_music['genre'] = merged_music['track_genre'].fillna(merged_music.get('genre', 'unknown')).fillna(
            'unknown')

    merged_music['album_name'] = merged_music.get('album_name', 'Unknown Album').fillna('Unknown Album')
    merged_music['popularity'] = merged_music.get('popularity', 0).fillna(0)
    merged_music['tempo'] = merged_music['tempo'].fillna(merged_music['tempo'].median())
    merged_music['lyrics'] = merged_music.get('lyrics', '').fillna('')

    # 5. 특성 스케일링 및 클러스터링
    x = merged_music[audio_features].copy()
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # K-Means 클러스터링 (5개 그룹)
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    merged_music['cluster'] = kmeans.fit_predict(x_scaled)

    # 6. 불필요한 컬럼 제거 및 중복 제거
    cols_to_drop = [c for c in merged_music.columns if '_tcc' in c or '_spot' in c or c == 'merge_key']
    merged_music = merged_music.drop(columns=cols_to_drop)

    if 'Unnamed: 0' in merged_music.columns:
        merged_music.drop(columns=['Unnamed: 0'], inplace=True)

    merged_music = merged_music.drop_duplicates(subset=['artist_name', 'track_name'], keep='first')

    # # 7. 최종 결과 저장
    # output_filename = 'final_preprocessed_music_data.csv'
    # merged_music.to_csv(output_filename, index=False, encoding='utf-8-sig')
    # print(f"성공적으로 저장되었습니다: {output_filename}")

    return merged_music


if __name__ == "__main__":
    final_df = preprocess_music_data()
