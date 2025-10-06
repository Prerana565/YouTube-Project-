import os
import re
import time
import pandas as pd
import isodate
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from functools import lru_cache
from collections import defaultdict

# Configuration
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"
YOUTUBE_API_KEY = "AIzaSyBozcvl25xP0tRAa1t6KSjn-hA-KKvVxMg"

# Helper functions
def extract_hashtags(description):
    return re.findall(r"#\w+", description) if description else []

def extract_keywords(text):
    text = re.sub(r"[^a-zA-Z\s]", "", str(text).lower())
    return [word for word in text.split() if len(word) > 2][:10]  # Top 10 keywords

def parse_duration(duration):
    try:
        return isodate.parse_duration(duration).total_seconds()
    except:
        return 0

@lru_cache(maxsize=100)
def get_category_name(category_id):
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=YOUTUBE_API_KEY)
    request = youtube.videoCategories().list(part="snippet", id=category_id)
    try:
        response = request.execute()
        return response['items'][0]['snippet']['title'] if response['items'] else "Unknown"
    except HttpError:
        return "Unknown"

# Core API handlers
def fetch_videos_by_order(topic, order_type, max_results=50):
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=YOUTUBE_API_KEY)

    video_data = []
    video_ids = []
    page_token = None

    try:
        while len(video_data) < max_results:
            request = youtube.search().list(
                q=topic,
                part="snippet",
                maxResults=min(50, max_results - len(video_data)),
                order=order_type,
                type="video",
                pageToken=page_token
            )
            response = request.execute()

            current_videos, current_ids = process_search_page(response, youtube)
            video_data.extend(current_videos)
            video_ids.extend(current_ids)

            page_token = response.get('nextPageToken')
            if not page_token:
                break

            time.sleep(1)  # Rate limit protection

    except HttpError as e:
        if e.resp.status == 403:
            raise Exception("YouTube API quota exceeded. Please try again later.")
        else:
            raise

    return video_data[:max_results], video_ids[:max_results]

def process_search_page(response, youtube):
    video_data = []
    video_ids = []

    for item in response.get('items', []):
        video_id = item['id']['videoId']
        
        stats_request = youtube.videos().list(
            part="statistics,contentDetails,snippet",
            id=video_id
        )
        try:
            stats_response = stats_request.execute()
            if not stats_response['items']:
                continue

            stats = stats_response['items'][0]
            duration = parse_duration(stats['contentDetails']['duration'])

            if duration < 60:
                continue  # Skip videos less than 1 minute

            video_info = parse_video_stats(stats)
            video_data.append(video_info)
            video_ids.append(video_id)

        except HttpError:
            continue

    return video_data, video_ids

def parse_video_stats(stats):
    snippet = stats.get('snippet', {})
    statistics = stats.get('statistics', {})
    content_details = stats.get('contentDetails', {})

    # Extract numbers
    views = int(statistics.get('viewCount', 0))
    likes = int(statistics.get('likeCount', 0))
    comments = int(statistics.get('commentCount', 0))
    engagement_score = (likes + comments) / views if views > 0 else 0

    return {
        "title": snippet.get("title", "Untitled"),
        "channel": snippet.get("channelTitle", "Unknown Channel"),
        "published_at": snippet.get("publishedAt", ""),
        "category": get_category_name(snippet.get('categoryId', '')),
        "tags": snippet.get('tags', [])[:5],
        "hashtags": extract_hashtags(snippet.get('description', '')),
        "title_keywords": extract_keywords(snippet.get('title', '')),
        "description_keywords": extract_keywords(snippet.get('description', '')),
        "views": views,
        "likes": likes,
        "comments": comments,
        "duration": parse_duration(content_details.get('duration', 'PT0S')),
        "engagement_score": engagement_score,
        "video_url": f"https://www.youtube.com/watch?v={stats['id']}"
    }

def fetch_top_comments(video_ids, max_comments=100):
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=YOUTUBE_API_KEY)
    comments = []

    for video_id in video_ids:
        if len(comments) >= max_comments:
            break

        try:
            comment_request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                order="relevance",
                maxResults=min(20, max_comments - len(comments))
            )
            comment_response = comment_request.execute()

            for item in comment_response.get('items', []):
                if 'topLevelComment' in item['snippet']:
                    comment = item['snippet']['topLevelComment']['snippet']
                    comments.append({
                        "video_id": video_id,
                        "comment": comment['textDisplay'],
                        "likes": comment.get('likeCount', 0),
                        "author": comment.get('authorDisplayName', 'Anonymous'),
                        "published_at": comment.get('publishedAt', '')
                    })

            time.sleep(0.5)  # Rate limit protection

        except HttpError as e:
            if e.resp.status == 403:
                continue  # Skip if comments are disabled
            else:
                raise

    return comments[:max_comments]

# Main fetcher with progress indicators
def fetch_all_data(topic, max_results=100):
    categories = {
        "recent": "date",
        "most_viewed": "viewCount",
        "trending": "relevance",
        "most_liked": "rating"
    }

    print("✅ Fetching video data...")
    combined_video_data = []
    categorized_comments = defaultdict(list)

    for category, order_type in categories.items():
        try:
            video_data, video_ids = fetch_videos_by_order(topic, order_type, max_results // 4)
            combined_video_data.extend(video_data)
            print(f"✅ {category.capitalize()} videos fetched")
            print("✅ Fetching comments...")
            categorized_comments[category] = fetch_top_comments(video_ids, max_comments=50)
        except Exception as e:
            print(f"❌ Error fetching {category} videos: {str(e)}")
            continue

    print("✅ Analyzing data...")
    video_df = pd.DataFrame(combined_video_data)

    comments_list = []
    for cat, comments in categorized_comments.items():
        for comment in comments:
            comments_list.append({
                "category": cat,
                "video_id": comment["video_id"],
                "comment": comment["comment"],
                "likes": comment["likes"],
                "author": comment["author"],
                "published_at": comment["published_at"]
            })

    comments_df = pd.DataFrame(comments_list)

    print("✅ Analysis complete.")

    # Print most engaging video
    if not video_df.empty:
        top_video = video_df.loc[video_df['engagement_score'].idxmax()]
        print("\n🔥 Most Engaging Video:")
        print(f"Title: {top_video['title']}")
        print(f"Engagement Score: {top_video['engagement_score']:.4f}")
        print(f"URL: {top_video['video_url']}")

    return video_df, comments_df
