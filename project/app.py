# app.py
import nltk
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import isodate
from datetime import datetime
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import google.generativeai as genai
from youtube_data_handler import fetch_all_data
from nltk.sentiment import SentimentIntensityAnalyzer
from keybert import KeyBERT
from collections import Counter
import google.generativeai as genai
from datetime import datetime
from PIL import Image
import io
import base64
import yt_dlp
import os
import tempfile
import bcrypt
import json
import requests
import re
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
 

# Preload NLTK Data
nltk.download(['punkt', 'wordnet', 'stopwords', 'vader_lexicon'], quiet=True)

# Configure APIs
genai.configure(api_key="AIzaSyCiWJwcl8u0tyOLYucXnA6JWlUwMMgvqbs")
YOUTUBE_API_KEY = "AIzaSyCqFyrK_QRFl1llBZ5TABF8N1ImFBQgNj4"

# Configure Streamlit
st.set_page_config(page_title="📊 YouTube Content Strategist", layout="wide")

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'landing'

# User management
def load_users():
    """Load users from file or create default"""
    try:
        with open('users.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Create default admin user with bcrypt
        import bcrypt
        hashed_password = bcrypt.hashpw('admin123'.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        default_users = {
            "admin": {
                "name": "Admin User",
                "password": hashed_password
            }
        }
        with open('users.json', 'w') as f:
            json.dump(default_users, f)
        return default_users

def save_users(users):
    """Save users to file"""
    with open('users.json', 'w') as f:
        json.dump(users, f)

def verify_password(password, hashed_password):
    """Verify password using bcrypt"""
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))

def hash_password(password):
    """Hash password using bcrypt"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def seconds_to_minutes(seconds):
    return seconds / 60 if seconds > 0 else 0

def calculate_engagement(row):
    views = row['views'] if row['views'] > 0 else 1
    return (2 * row['likes'] + row['comments']) / views

def generate_content_pdf(topic, analysis_type, df, summary_metrics, rec_text=None, images=None):
    """Generate a concise PDF report for Advanced Content Analysis and return bytes."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    margin = 40
    y = height - margin

    def write_line(text, font="Helvetica", size=10, leading=14):
        nonlocal y
        c.setFont(font, size)
        wrapped = []
        # Simple wrap for long lines
        max_chars = 95
        for i in range(0, len(text), max_chars):
            wrapped.append(text[i:i+max_chars])
        for line in wrapped:
            c.drawString(margin, y, line)
            y -= leading
            if y < margin:
                c.showPage()
                y = height - margin

    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, "YT Brain — Advanced YouTube Content Analysis Report")
    y -= 24
    c.setFont("Helvetica", 11)
    c.drawString(margin, y, f"Topic: {topic}")
    y -= 14
    c.drawString(margin, y, f"Analysis Type: {analysis_type}")
    y -= 20

    # Summary metrics
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Summary Metrics")
    y -= 16
    c.setFont("Helvetica", 10)
    for k, v in summary_metrics.items():
        write_line(f"- {k}: {v}")
    y -= 8

    # Top videos table (by engagement)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Top Performing Videos (by engagement)")
    y -= 16
    c.setFont("Helvetica", 10)
    top_cols = [col for col in ["title", "channel", "engagement", "views"] if col in df.columns]
    if not df.empty and top_cols:
        top = df[df["engagement"].notna()] if "engagement" in df.columns else df
        top = top.nlargest(min(5, len(top)), "engagement") if "engagement" in top.columns else top.head(5)
        for _, row in top.iterrows():
            title = str(row.get("title", "N/A"))[:60]
            channel = str(row.get("channel", "N/A"))[:30]
            engagement = row.get("engagement", 0)
            views = row.get("views", 0)
            write_line(f"• {title} | {channel} | Engagement: {engagement:.2f}% | Views: {int(views):,}")
    else:
        write_line("No data available")

    # AI Recommendations section
    if rec_text:
        y -= 12
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y, "AI Strategy Recommendations")
        y -= 16
        c.setFont("Helvetica", 10)
        for paragraph in rec_text.split("\n"):
            if paragraph.strip() == "":
                y -= 6
                continue
            write_line(paragraph)

    # Visualizations section
    if images:
        y -= 12
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y, "Advanced Data Visualizations")
        y -= 16
        max_img_width = width - 2 * margin
        for img_bytes in images:
            try:
                img = ImageReader(io.BytesIO(img_bytes))
                iw, ih = img.getSize()
                scale = min(max_img_width / iw, (height - 2 * margin) / ih)
                draw_w, draw_h = iw * scale, ih * scale
                if y - draw_h < margin:
                    c.showPage()
                    y = height - margin
                c.drawImage(img, margin, y - draw_h, width=draw_w, height=draw_h, preserveAspectRatio=True, anchor='sw')
                y -= (draw_h + 14)
                if y < margin:
                    c.showPage()
                    y = height - margin
            except Exception:
                continue

    c.showPage()
    c.save()
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

def generate_url_pdf_report(analysis_url, analysis_depth, video_data, extra_text=None, images=None):
    """Generate a PDF report for single URL analysis and return bytes."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    margin = 40
    y = height - margin

    def write_line(text, font="Helvetica", size=10, leading=14):
        nonlocal y
        c.setFont(font, size)
        max_chars = 95
        wrapped = [text[i:i+max_chars] for i in range(0, len(text), max_chars)] if text else [""]
        for line in wrapped:
            c.drawString(margin, y, line)
            y -= leading
            if y < margin:
                c.showPage()
                y = height - margin

    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, "YT Brain — YouTube URL Analysis Report")
    y -= 24
    c.setFont("Helvetica", 11)
    c.drawString(margin, y, f"URL: {analysis_url}")
    y -= 14
    c.drawString(margin, y, f"Analysis Depth: {analysis_depth}")
    y -= 20

    # Performance metrics
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Performance Metrics")
    y -= 16
    c.setFont("Helvetica", 10)
    write_line(f"Title: {video_data.get('title','N/A')}")
    write_line(f"Channel: {video_data.get('channel','N/A')}")
    write_line(f"Views: {video_data.get('views',0):,}")
    write_line(f"Likes: {video_data.get('likes',0):,}")
    write_line(f"Comments: {video_data.get('comments',0):,}")
    write_line(f"Engagement Rate: {video_data.get('engagement_rate',0):.2f}%")
    write_line(f"Upload Date: {video_data.get('upload_date','')}")
    write_line(f"Duration: {video_data.get('duration_mins',0):.1f} minutes")
    y -= 8

    # Content Analysis
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Content Analysis")
    y -= 16
    c.setFont("Helvetica", 10)
    title_kw = ", ".join(video_data.get('title_keywords', [])[:15])
    desc_kw = ", ".join(video_data.get('description_keywords', [])[:15])
    hashtags = video_data.get('hashtags', [])
    write_line(f"Title Keywords: {title_kw}")
    write_line(f"Description Keywords: {desc_kw}")
    write_line(f"Hashtags: {' '.join(hashtags[:15])}")

    # Recommendations
    recs = video_data.get('recommendations', [])
    if recs:
        y -= 8
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y, "Optimization Recommendations")
        y -= 16
        c.setFont("Helvetica", 10)
        for rec in recs[:20]:
            write_line(f"- {rec}")

    # Extra text (AI insights) if provided
    if extra_text:
        y -= 8
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y, "Additional Insights")
        y -= 16
        c.setFont("Helvetica", 10)
        for paragraph in extra_text.split('\n'):
            if paragraph.strip() == "":
                y -= 6
                continue
            write_line(paragraph)

    # Visuals (if any were captured)
    if images:
        y -= 12
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y, "Visualizations")
        y -= 16
        max_img_width = width - 2 * margin
        for img_bytes in images:
            try:
                img = ImageReader(io.BytesIO(img_bytes))
                iw, ih = img.getSize()
                scale = min(max_img_width / iw, (height - 2 * margin) / ih)
                draw_w, draw_h = iw * scale, ih * scale
                if y - draw_h < margin:
                    c.showPage()
                    y = height - margin
                c.drawImage(img, margin, y - draw_h, width=draw_w, height=draw_h, preserveAspectRatio=True, anchor='sw')
                y -= (draw_h + 14)
                if y < margin:
                    c.showPage()
                    y = height - margin
            except Exception:
                continue

    c.showPage()
    c.save()
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

def download_audio_from_url(url, output_path):
    """Download audio from YouTube URL"""
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': output_path + '.%(ext)s',
            'ignoreerrors': True,
            'no_warnings': True,
            'noplaylist': True,
            'skip_download': False,
            'quiet': True,
            'geo_bypass': True,
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'
            },
            'extractor_args': {
                'youtube': {
                    'player_client': ['android', 'web']
                }
            },
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return True
    except Exception as e:
        st.error(f"Error downloading audio: {str(e)}")
        return False

def audio_download_page():
    st.header("**📊 URL Analysis**")
    st.markdown("Get detailed insights about any YouTube video including performance metrics, content analysis, and optimization recommendations.")
    
    # Analysis options
    col1, col2 = st.columns(2)
    with col1:
        analysis_url = st.text_input("Enter YouTube URL for Analysis:", placeholder="https://www.youtube.com/watch?v=...", key="main_comprehensive_analysis_url_3")
    with col2:
        analysis_depth = st.selectbox(
            "Analysis Depth:",
            ["Basic Analysis", "Full Analysis with Recommendations"],
            key="main_analysis_depth_selector_3"
        )
    
    if st.button("🔍 Analyze Video", type="primary", key="main_comprehensive_analyze_button_4"):
        if analysis_url:
            with st.spinner("Performing comprehensive analysis..."):
                try:
                    # Validate URL format before analysis
                    if not ('youtube.com' in analysis_url or 'youtu.be' in analysis_url):
                        st.error("❌ Please enter a valid YouTube URL (youtube.com or youtu.be)")
                        return
                    
                    video_data = analyze_single_video(analysis_url)
                    
                    if video_data['success']:
                        st.success(f"✅ Analysis Complete: {video_data['title'][:60]}...")
                        url_images_for_pdf = []
                        
                        # Performance Metrics Dashboard
                        st.markdown("### 📈 Performance Metrics")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Views", f"{video_data['views']:,}", 
                                     delta=f"{video_data['views']/1000:.1f}K" if video_data['views'] > 1000 else "")
                        with col2:
                            st.metric("Likes", f"{video_data['likes']:,}", 
                                     delta=f"{video_data['likes']/1000:.1f}K" if video_data['likes'] > 1000 else "")
                        with col3:
                            st.metric("Comments", f"{video_data.get('comments', 0):,}")
                        with col4:
                            st.metric("Engagement Rate", f"{video_data['engagement_rate']:.2f}%", 
                                     delta="High" if video_data['engagement_rate'] > 2 else "Low")
                        
                        # Detailed Video Information
                        st.markdown("### 📋 Video Information")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"""
                            **🎬 Title:** {video_data['title']}
                            
                            **📺 Channel:** {video_data['channel']}
                            
                            **📅 Upload Date:** {video_data['upload_date']}
                            
                            **⏱️ Duration:** {video_data['duration_mins']:.1f} minutes
                            
                            **💬 Comments:** {video_data.get('comments', 0):,}
                            """)
                        with col2:
                            st.markdown(f"""
                            **🔊 Audio Quality:** {video_data['audio_bitrate']} kbps
                            
                            **😊 Sentiment:** {video_data['viewer_sentiment']}
                            
                            **📊 Sentiment Score:** {video_data['sentiment_score']:.2f}/1.0
                            
                            **🎵 Music Score:** {video_data['music_score']}/10
                            """)
                        
                        # Content Analysis
                        if analysis_depth in ["Detailed Analysis", "Full Analysis with Recommendations"]:
                            st.markdown("### 🔍 Content Analysis")
                            
                            # Engagement Analysis
                            engagement_status = "Excellent" if video_data['engagement_rate'] > 3 else \
                                               "Good" if video_data['engagement_rate'] > 1.5 else \
                                               "Average" if video_data['engagement_rate'] > 0.5 else "Low"
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.markdown(f"""
                                **📊 Engagement Analysis:**
                                - Status: {engagement_status}
                                - Rate: {video_data['engagement_rate']:.2f}%
                                - Performance: {'Above Average' if video_data['engagement_rate'] > 1.5 else 'Below Average'}
                                """)
                            
                            with col2:
                                st.markdown(f"""
                                **🎵 Audio Analysis:**
                                - Quality: {'High' if video_data['audio_bitrate'] > 128 else 'Medium' if video_data['audio_bitrate'] > 64 else 'Low'}
                                - Bitrate: {video_data['audio_bitrate']} kbps
                                - Music Focus: {'Yes' if video_data['music_score'] > 3 else 'No'}
                                """)
                            
                            with col3:
                                st.markdown(f"""
                                **😊 Sentiment Analysis:**
                                - Overall: {video_data['viewer_sentiment']}
                                - Score: {video_data['sentiment_score']:.2f}/1.0
                                - Positive Indicators: {video_data['positive_feedback']}
                                - Negative Indicators: {video_data['negative_feedback']}
                                """)
                        
                        # Optimization Recommendations
                        if analysis_depth == "Full Analysis with Recommendations":
                            st.markdown("### 💡 Optimization Recommendations")
                            
                            recommendations = []
                            
                            # Engagement recommendations
                            if video_data['engagement_rate'] < 1.5:
                                recommendations.append("🎯 **Improve Engagement:** Focus on creating more interactive content, ask questions, and encourage viewer participation")
                            
                            # Duration recommendations
                            if video_data['duration_mins'] < 3:
                                recommendations.append("⏱️ **Consider Longer Content:** Videos under 3 minutes may not provide enough value for viewers")
                            elif video_data['duration_mins'] > 20:
                                recommendations.append("⏱️ **Optimize Length:** Consider breaking longer content into series for better retention")
                            
                            # Audio recommendations
                            if video_data['audio_bitrate'] < 128:
                                recommendations.append("🔊 **Improve Audio Quality:** Higher audio bitrate can enhance viewer experience")
                            
                            # Music recommendations
                            if video_data['music_score'] < 3:
                                recommendations.append("🎵 **Add Background Music:** Music can improve engagement and viewer retention")
                            
                            # Sentiment recommendations
                            if video_data['sentiment_score'] < 0.5:
                                recommendations.append("😊 **Improve Content Tone:** Focus on more positive and engaging content")
                            
                            if recommendations:
                                for rec in recommendations:
                                    st.markdown(f"- {rec}")
                            else:
                                st.success("🎉 Your video appears to be well-optimized!")

                            # store for PDF
                            video_data['recommendations'] = recommendations
                        
                        # Additional Analysis Features
                        if analysis_depth == "Full Analysis with Recommendations":
                            st.markdown("### 📈 Advanced Insights")
                            
                            # Performance Prediction
                            st.markdown("#### 🎯 Performance Prediction")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                # Calculate potential reach based on engagement
                                potential_reach = video_data['views'] * (1 + video_data['engagement_rate'] / 100)
                                st.metric("Potential Reach", f"{potential_reach:,.0f}", 
                                         delta=f"+{potential_reach - video_data['views']:,.0f}")
                            
                            with col2:
                                # Calculate optimal posting time (simplified)
                                optimal_time = "3-5 PM" if video_data['engagement_rate'] > 2 else "7-9 PM"
                                st.metric("Optimal Posting Time", optimal_time)
                            
                            with col3:
                                # Calculate content score
                                content_score = min(100, (video_data['engagement_rate'] * 20 + 
                                                         video_data['sentiment_score'] * 30 + 
                                                         min(video_data['audio_bitrate'] / 2, 20)))
                                st.metric("Content Score", f"{content_score:.0f}/100")

                            # Capture a simple metrics bar chart for PDF
                            try:
                                fig, ax = plt.subplots(figsize=(6, 4))
                                metrics_vals = [video_data['views'], video_data['likes'], video_data.get('comments', 0)]
                                labels = ['Views', 'Likes', 'Comments']
                                colors = ['#45b7d1', '#4ecdc4', '#ff6b6b']
                                ax.bar(labels, metrics_vals, color=colors)
                                ax.set_title('Key Metrics')
                                ax.set_ylabel('Count')
                                buf = io.BytesIO()
                                fig.savefig(buf, format='png', bbox_inches='tight')
                                url_images_for_pdf.append(buf.getvalue())
                                buf.close()
                                plt.close()
                            except Exception:
                                pass
                            
                            # Hashtag Recommendations
                            st.markdown("#### #️⃣ Recommended Hashtags")
                            
                            # Generate hashtags based on content analysis
                            hashtags = []
                            title_lower = video_data['title'].lower()
                            
                            # Content-based hashtags
                            if 'music' in title_lower or video_data['music_score'] > 3:
                                hashtags.extend(['#music', '#song', '#audio', '#musicvideo', '#newmusic'])
                            if 'tutorial' in title_lower or 'how to' in title_lower:
                                hashtags.extend(['#tutorial', '#howto', '#tips', '#learning', '#education'])
                            if 'review' in title_lower:
                                hashtags.extend(['#review', '#productreview', '#honestreview', '#recommendation'])
                            if 'gaming' in title_lower or 'game' in title_lower:
                                hashtags.extend(['#gaming', '#gamer', '#videogames', '#streamer', '#gamingcommunity'])
                            
                            # Performance-based hashtags
                            if video_data['engagement_rate'] > 2:
                                hashtags.extend(['#viral', '#trending', '#popular', '#engaging', '#quality'])
                            if video_data['views'] > 10000:
                                hashtags.extend(['#success', '#hit', '#amazing', '#awesome'])
                            
                            # General viral hashtags
                            hashtags.extend(['#youtube', '#video', '#content', '#creator', '#subscribe', '#like', '#share'])
                            
                            # Display hashtags in columns
                            col1, col2, col3 = st.columns(3)
                            unique_hashtags = list(set(hashtags))[:15]  # Limit to 15 unique hashtags
                            
                            with col1:
                                for i, hashtag in enumerate(unique_hashtags[:5]):
                                    st.markdown(f"- {hashtag}")
                            with col2:
                                for i, hashtag in enumerate(unique_hashtags[5:10]):
                                    st.markdown(f"- {hashtag}")
                            with col3:
                                for i, hashtag in enumerate(unique_hashtags[10:15]):
                                    st.markdown(f"- {hashtag}")
                            
                            # Copy hashtags button
                            hashtag_string = ' '.join(unique_hashtags)
                            st.code(hashtag_string, language="text")
                            
                            # Trending Analysis
                            st.markdown("#### 📊 Trending Analysis")
                            
                            # Simulate trending analysis based on video metrics
                            trending_score = (video_data['engagement_rate'] * 0.4 + 
                                             video_data['sentiment_score'] * 0.3 + 
                                             min(video_data['views'] / 10000, 1) * 0.3)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"""
                                **📈 Trending Potential:**
                                - Score: {trending_score:.1f}/1.0
                                - Status: {'🔥 Trending' if trending_score > 0.7 else '📈 Growing' if trending_score > 0.5 else '📊 Stable'}
                                - Viral Chance: {'High' if trending_score > 0.8 else 'Medium' if trending_score > 0.6 else 'Low'}
                                """)
                            
                            with col2:
                                st.markdown(f"""
                                **🎯 Growth Opportunities:**
                                - Audience Retention: {'Excellent' if video_data['engagement_rate'] > 3 else 'Good' if video_data['engagement_rate'] > 1.5 else 'Needs Improvement'}
                                - Content Quality: {'High' if video_data['sentiment_score'] > 0.7 else 'Medium' if video_data['sentiment_score'] > 0.5 else 'Low'}
                                - Discoverability: {'High' if video_data['views'] > 50000 else 'Medium' if video_data['views'] > 10000 else 'Low'}
                                """)
                            
                            # Performance Timeline
                            st.markdown("#### ⏰ Performance Timeline")
                            
                            # Create a simple performance timeline
                            timeline_data = {
                                'Day 1': video_data['views'] * 0.1,
                                'Week 1': video_data['views'] * 0.3,
                                'Month 1': video_data['views'] * 0.6,
                                'Month 3': video_data['views'] * 0.8,
                                'Month 6': video_data['views']
                            }
                            
                            col1, col2, col3, col4, col5 = st.columns(5)
                            for i, (period, views) in enumerate(timeline_data.items()):
                                with [col1, col2, col3, col4, col5][i]:
                                    st.metric(period, f"{views:,.0f}")
                            
                            # SEO Recommendations
                            st.markdown("#### 🔍 SEO & Discovery Tips")
                            
                            seo_tips = []
                            
                            # Title optimization
                            if len(video_data['title']) < 30:
                                seo_tips.append("📝 **Title Length:** Consider making your title longer (30-60 characters) for better SEO")
                            elif len(video_data['title']) > 70:
                                seo_tips.append("📝 **Title Length:** Your title might be too long. Keep it under 70 characters")
                            
                            # Description optimization
                            if video_data['music_score'] > 3:
                                seo_tips.append("🎵 **Music Content:** Add music-related keywords to your description for better discoverability")
                            
                            # Engagement optimization
                            if video_data['engagement_rate'] < 1:
                                seo_tips.append("👥 **Engagement:** Add calls-to-action in your video to improve engagement rates")
                            
                            # Duration optimization
                            if 5 <= video_data['duration_mins'] <= 15:
                                seo_tips.append("⏱️ **Duration:** Your video length is optimal for YouTube's algorithm")
                            else:
                                seo_tips.append("⏱️ **Duration:** Consider optimizing video length for better retention")
                            
                            for tip in seo_tips:
                                st.markdown(f"- {tip}")
                            
                            # Competitor Analysis
                            st.markdown("#### 🏆 Competitive Analysis")
                            
                            # Simulate competitor analysis
                            competitor_metrics = {
                                'Your Video': {
                                    'views': video_data['views'],
                                    'engagement': video_data['engagement_rate'],
                                    'duration': video_data['duration_mins']
                                },
                                'Average in Niche': {
                                    'views': video_data['views'] * 0.8,
                                    'engagement': 1.5,
                                    'duration': 8.5
                                },
                                'Top Performers': {
                                    'views': video_data['views'] * 2.5,
                                    'engagement': 3.2,
                                    'duration': 12.0
                                }
                            }
                            
                            col1, col2, col3 = st.columns(3)
                            for i, (category, metrics) in enumerate(competitor_metrics.items()):
                                with [col1, col2, col3][i]:
                                    st.markdown(f"**{category}**")
                                    st.markdown(f"- Views: {metrics['views']:,.0f}")
                                    st.markdown(f"- Engagement: {metrics['engagement']:.1f}%")
                                    st.markdown(f"- Duration: {metrics['duration']:.1f} min")
                            
                            # Action Plan
                            st.markdown("#### 📋 Action Plan")
                            
                            action_items = []
                            
                            if video_data['engagement_rate'] < 1.5:
                                action_items.append("🎯 **Immediate:** Add more interactive elements to your next video")
                            
                            if video_data['audio_bitrate'] < 128:
                                action_items.append("🔊 **Short-term:** Invest in better audio equipment")
                            
                            if video_data['sentiment_score'] < 0.6:
                                action_items.append("😊 **Content:** Focus on more positive and engaging content")
                            
                            if video_data['views'] < 1000:
                                action_items.append("📈 **Growth:** Implement the recommended hashtags and SEO tips")
                            
                            if not action_items:
                                action_items.append("🎉 **Maintain:** Keep up the excellent work! Your content is performing well.")
                            
                            for i, action in enumerate(action_items, 1):
                                st.markdown(f"{i}. {action}")

                            # Provide PDF download for URL analysis
                            try:
                                pdf_bytes = generate_url_pdf_report(analysis_url, analysis_depth, video_data, images=url_images_for_pdf)
                                st.download_button(
                                    label="⬇️ Download PDF Report",
                                    data=pdf_bytes,
                                    file_name="yt_url_analysis_report.pdf",
                                    mime="application/pdf",
                                    key="url_analysis_pdf_download"
                                )
                            except Exception:
                                pass
                        
                    else:
                        st.error(f"❌ Analysis failed: {video_data['error']}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("Please enter a YouTube URL")
    
    # Instructions
    with st.expander("💡 How to Use URL Analysis"):
        st.markdown("""
        **Basic Analysis:** Get essential metrics and video information
        
        **Detailed Analysis:** Includes content analysis, engagement insights, and audio quality assessment
        
        **Full Analysis:** Provides optimization recommendations and actionable insights
        
        **Features:**
        - 📊 Performance metrics dashboard
        - 🔍 Content quality analysis
        - 💡 Optimization recommendations
        - 🚀 Quick action buttons
        - 📈 Engagement insights
        - 🎵 Audio quality assessment
        """)

def extract_video_comments(url, max_comments=50):
    """Extract comments from a YouTube video"""
    try:
        # Validate URL
        if not url or not isinstance(url, str):
            raise ValueError("Invalid URL provided")
        
        url = url.strip()
        if not ('youtube.com' in url or 'youtu.be' in url):
            raise ValueError("Invalid YouTube URL")
        
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'writesubtitles': False,
            'writeautomaticsub': False,
            'ignoreerrors': True,
            'skip_download': True,
            'noplaylist': True,
            'geo_bypass': True,
            'format': 'best[height<=720]/best',
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'
            },
            'extractor_args': {
                'youtube': {
                    'player_client': ['android', 'web']
                }
            },
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            
            # Note: yt-dlp doesn't extract comments by default
            # We'll simulate comment analysis with sentiment indicators
            comments = []
            description = info.get('description', '')
            
            # Extract basic feedback indicators from description and title
            title = info.get('title', '').lower()
            desc_lower = description.lower()
            
            # Analyze sentiment indicators with improved detection
            positive_words = ['amazing', 'great', 'awesome', 'love', 'best', 'perfect', 'excellent', 'good', 'nice', 'wonderful', 'fantastic', 'incredible', 'outstanding']
            negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'boring', 'poor', 'disappointing', 'horrible', 'useless', 'waste']
            
            # Count sentiment words more accurately
            positive_count = 0
            negative_count = 0
            
            combined_text = (title + ' ' + desc_lower).lower()
            for word in positive_words:
                positive_count += combined_text.count(word)
            for word in negative_words:
                negative_count += combined_text.count(word)
            
            # Generate feedback based on video metrics with improved logic
            view_count = info.get('view_count', 0)
            like_count = info.get('like_count', 0)
            comment_count = info.get('comment_count', 0)
            
            # Safely convert to integers
            try:
                view_count = int(view_count) if view_count else 0
                like_count = int(like_count) if like_count else 0
                comment_count = int(comment_count) if comment_count else 0
            except (ValueError, TypeError):
                view_count = like_count = comment_count = 0
            
            if view_count > 0:
                # Calculate engagement including comments
                total_engagement = like_count + comment_count
                engagement_rate = (total_engagement / view_count) * 100
                
                # Determine sentiment based on engagement and content analysis
                base_sentiment_score = 0.5
                
                if engagement_rate > 3:
                    sentiment = "Very Positive"
                    base_sentiment_score = 0.8
                elif engagement_rate > 1.5:
                    sentiment = "Positive"
                    base_sentiment_score = 0.7
                elif engagement_rate > 0.8:
                    sentiment = "Neutral"
                    base_sentiment_score = 0.5
                else:
                    sentiment = "Mixed"
                    base_sentiment_score = 0.4
                
                # Adjust based on content sentiment
                sentiment_adjustment = (positive_count * 0.05) - (negative_count * 0.03)
                sentiment_score = max(0, min(1, base_sentiment_score + sentiment_adjustment))
            else:
                sentiment = "Unknown"
                sentiment_score = 0.5
                engagement_rate = 0
            
            return {
                'sentiment': sentiment,
                'sentiment_score': round(min(max(sentiment_score, 0), 1), 3),
                'positive_indicators': positive_count,
                'negative_indicators': negative_count,
                'engagement_rate': round(engagement_rate, 3) if 'engagement_rate' in locals() else 0
            }
    
    except Exception as e:
        return {
            'sentiment': 'Error',
            'sentiment_score': 0.5,
            'positive_indicators': 0,
            'negative_indicators': 0,
            'engagement_rate': 0
        }

def analyze_single_video(url):
    """Analyze a single video and return comprehensive data"""
    try:
        # Validate URL first
        if not url or not isinstance(url, str):
            raise ValueError("Invalid URL provided")
        
        # Clean the URL
        url = url.strip()
        if not ('youtube.com' in url or 'youtu.be' in url):
            raise ValueError("Please provide a valid YouTube URL")
        
        # Single, robust extraction strategy
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'ignoreerrors': False,  # We want to catch errors
            'skip_download': True,
            'no_check_certificate': True,
            'noplaylist': True,
            'cachedir': False,
            'geo_bypass': True,
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'
            },
            'extractor_args': {
                'youtube': {
                    'player_client': ['android', 'web']
                }
            },
        }
        
        info = None
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                # If a playlist-like structure is returned, pick the first valid entry
                if info and isinstance(info, dict) and info.get('entries'):
                    entries = [e for e in info.get('entries', []) if e]
                    info = entries[0] if entries else None
        except Exception as e:
            # If extraction fails, try with minimal options
            minimal_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': True,
                'ignoreerrors': True,
                'skip_download': True,
            }
            try:
                with yt_dlp.YoutubeDL(minimal_opts) as ydl:
                    info = ydl.extract_info(url, download=False)
                    if info and isinstance(info, dict) and info.get('entries'):
                        entries = [e for e in info.get('entries', []) if e]
                        info = entries[0] if entries else None
            except Exception:
                # Last resort: oEmbed fallback to at least get title/author
                try:
                    resp = requests.get(
                        "https://www.youtube.com/oembed",
                        params={"url": url, "format": "json"},
                        timeout=8,
                    )
                    if resp.ok:
                        data = resp.json()
                        # Build a minimal info structure
                        info = {
                            'title': data.get('title', 'Unknown Video'),
                            'uploader': data.get('author_name', 'Unknown Channel'),
                            'description': '',
                            'upload_date': '',
                            'duration': 0,
                            'view_count': 0,
                            'like_count': 0,
                            'comment_count': 0,
                            'formats': [],
                            '_oembed_fallback': True
                        }
                    else:
                        raise Exception(f"Failed to extract video information: {str(e)}")
                except Exception:
                    raise Exception(f"Failed to extract video information: {str(e)}")
        
        # Helper: extract video id from url
        def _extract_video_id(u: str, fallback: str = "") -> str:
            try:
                # youtu.be/VIDEOID
                m = re.search(r"youtu\.be/([A-Za-z0-9_-]{6,})", u)
                if m:
                    return m.group(1)
                # youtube.com/watch?v=VIDEOID
                m = re.search(r"[?&]v=([A-Za-z0-9_-]{6,})", u)
                if m:
                    return m.group(1)
                # shorts
                m = re.search(r"/shorts/([A-Za-z0-9_-]{6,})", u)
                if m:
                    return m.group(1)
                return fallback
            except Exception:
                return fallback

        # Helper: fetch stats via YouTube Data API v3
        def _fetch_stats_via_api(video_id: str):
            try:
                if not video_id:
                    return None
                youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
                resp = youtube.videos().list(part='statistics,snippet,contentDetails', id=video_id).execute()
                items = resp.get('items') or []
                if not items:
                    return None
                stats = items[0].get('statistics', {})
                snippet = items[0].get('snippet', {})
                content = items[0].get('contentDetails', {})
                # Parse ISO 8601 duration if needed
                duration_seconds = 0
                try:
                    dur = content.get('duration')
                    if dur:
                        duration_seconds = int(isodate.parse_duration(dur).total_seconds())
                except Exception:
                    duration_seconds = 0
                return {
                    'title': snippet.get('title'),
                    'channel': snippet.get('channelTitle'),
                    'upload_date': snippet.get('publishedAt', '')[:10].replace('T',' '),
                    'view_count': int(stats.get('viewCount', 0) or 0),
                    'like_count': int(stats.get('likeCount', 0) or 0),
                    'comment_count': int(stats.get('commentCount', 0) or 0),
                    'duration': duration_seconds,
                }
            except Exception:
                return None

        if info:
            # Extract basic information with robust fallbacks
            title = str(info.get('title') or info.get('fulltitle') or 'Unknown Video')
            uploader = str(info.get('uploader') or info.get('channel') or info.get('uploader_id') or 'Unknown Channel')
            description = str(info.get('description') or '')
            upload_date = str(info.get('upload_date') or '')
            
            # Safely extract numeric values with validation
            def safe_int(value, default=0):
                try:
                    if value is None:
                        return default
                    return int(float(str(value)))
                except (ValueError, TypeError):
                    return default
            
            duration = safe_int(info.get('duration'), 0)
            view_count = safe_int(info.get('view_count'), 0)
            like_count = safe_int(info.get('like_count'), 0)
            comment_count = safe_int(info.get('comment_count'), 0)

            # If high-profile videos don't return counts via yt-dlp, fetch via API as fallback
            need_api = (view_count == 0) or (like_count == 0 and comment_count == 0)
            video_id = info.get('id') or _extract_video_id(url)
            if need_api and video_id:
                api_stats = _fetch_stats_via_api(video_id)
                if api_stats:
                    # Prefer API stats when greater than existing
                    if api_stats.get('title') and title == 'Unknown Video':
                        title = api_stats['title']
                    if api_stats.get('channel') and uploader == 'Unknown Channel':
                        uploader = api_stats['channel']
                    if api_stats.get('upload_date'):
                        upload_date = api_stats['upload_date']
                    view_count = max(view_count, api_stats.get('view_count', 0))
                    like_count = max(like_count, api_stats.get('like_count', 0))
                    comment_count = max(comment_count, api_stats.get('comment_count', 0))
                    duration = max(duration, api_stats.get('duration', 0))
            
            # Calculate metrics with safety checks
            duration_mins = duration / 60 if duration > 0 else 0
            # Engagement: compute from available metrics only
            engagement_rate = 0.0
            if view_count > 0:
                numerator = 0
                if like_count > 0:
                    numerator += like_count
                if comment_count > 0:
                    numerator += comment_count
                if numerator > 0:
                    engagement_rate = (numerator / view_count) * 100
            
            # Audio quality analysis with improved error handling
            max_audio_bitrate = 128  # Default fallback
            try:
                formats = info.get('formats', [])
                if isinstance(formats, list) and formats:
                    audio_bitrates = []
                    for f in formats:
                        if isinstance(f, dict):
                            acodec = f.get('acodec')
                            abr = f.get('abr')
                            if acodec and acodec != 'none' and abr:
                                try:
                                    audio_bitrates.append(int(float(abr)))
                                except (ValueError, TypeError):
                                    continue
                    if audio_bitrates:
                        max_audio_bitrate = max(audio_bitrates)
            except Exception:
                pass  # Use default value
            
            # Content analysis with improved keyword detection
            title_lower = title.lower() if title else ''
            desc_lower = description.lower() if description else ''
            
            music_keywords = ['music', 'song', 'audio', 'instrumental', 'beats', 'melody', 'soundtrack', 'cover', 'remix']
            music_score = sum(1 for word in music_keywords if word in title_lower or word in desc_lower)
            music_score = min(music_score, 10)  # Cap at 10
            
            # Get viewer feedback with improved error handling
            feedback = {
                'sentiment': 'Unknown',
                'sentiment_score': 0.5,
                'positive_indicators': 0,
                'negative_indicators': 0
            }
            
            try:
                feedback = extract_video_comments(url)
                # Validate feedback data
                if not isinstance(feedback, dict):
                    feedback = {
                        'sentiment': 'Unknown',
                        'sentiment_score': 0.5,
                        'positive_indicators': 0,
                        'negative_indicators': 0
                    }
            except Exception:
                pass  # Use default feedback
            
            # Format upload date for display
            formatted_date = upload_date
            if upload_date and len(upload_date) == 8:  # Format: YYYYMMDD
                try:
                    formatted_date = f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:8]}"
                except:
                    pass
            
            return {
                'url': url,
                'title': title[:200],  # Limit title length
                'channel': uploader[:100],  # Limit channel name length
                'duration_mins': round(duration_mins, 2),
                'views': view_count,
                'likes': like_count,
                'comments': comment_count,
                'engagement_rate': round(engagement_rate, 3),
                'upload_date': formatted_date,
                'music_score': music_score,
                'audio_bitrate': max_audio_bitrate,
                'viewer_sentiment': feedback.get('sentiment', 'Unknown'),
                'sentiment_score': round(feedback.get('sentiment_score', 0.5), 3),
                'positive_feedback': feedback.get('positive_indicators', 0),
                'negative_feedback': feedback.get('negative_indicators', 0),
                'success': True
            }
        else:
            # Return minimal data structure for failed extractions
            return {
                'url': url,
                'title': 'Video Analysis Failed',
                'channel': 'Unknown',
                'duration_mins': 0,
                'views': 0,
                'likes': 0,
                'comments': 0,
                'engagement_rate': 0,
                'upload_date': '',
                'music_score': 0,
                'audio_bitrate': 128,
                'viewer_sentiment': 'Unknown',
                'sentiment_score': 0.5,
                'positive_feedback': 0,
                'negative_feedback': 0,
                'success': False,
                'error': 'Could not extract video information - video may be private, deleted, or restricted'
            }
    
    except Exception as e:
        # Return minimal data structure for exceptions
        error_msg = str(e)
        if "HTTP Error 429" in error_msg:
            error_msg = "Rate limit exceeded. Please try again in a few minutes."
        elif "HTTP Error 403" in error_msg:
            error_msg = "Access denied. Video may be private or restricted."
        elif "HTTP Error 404" in error_msg:
            error_msg = "Video not found. Please check the URL."
        elif "network" in error_msg.lower() or "connection" in error_msg.lower():
            error_msg = "Network connection error. Please check your internet connection."
        
        return {
            'url': url if url else '',
            'title': 'Video Analysis Error',
            'channel': 'Unknown',
            'duration_mins': 0,
            'views': 0,
            'likes': 0,
            'comments': 0,
            'engagement_rate': 0,
            'upload_date': '',
            'music_score': 0,
            'audio_bitrate': 128,
            'viewer_sentiment': 'Unknown',
            'sentiment_score': 0.5,
            'positive_feedback': 0,
            'negative_feedback': 0,
            'success': False,
            'error': error_msg
        }

def landing_page():
    st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <h1 style="color: #FF6B6B; font-size: 3rem; margin-bottom: 1rem;">🎥 YT Brain</h1>
        <h2 style="color: #4ECDC4; font-size: 2rem; margin-bottom: 2rem;">The AI Toolkit for YouTube Success</h2>
        <p style="font-size: 1.2rem; color: #666; margin-bottom: 3rem;">
            Transform your YouTube content with AI-powered insights, thumbnail generation, and strategic recommendations.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; border: 2px solid #FF6B6B; border-radius: 10px; margin: 1rem;">
            <h3>📊 Content Analysis</h3>
            <p>Analyze trending videos and audience engagement patterns</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; border: 2px solid #4ECDC4; border-radius: 10px; margin: 1rem;">
            <h3>🎨 Thumbnail Studio</h3>
            <p>Generate professional thumbnails with AI assistance</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; border: 2px solid #45B7D1; border-radius: 10px; margin: 1rem;">
            <h3>📊 URL Analysis</h3>
            <p>Analyze YouTube videos for performance insights and optimization</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-top: 3rem;">
        <h3>Ready to boost your YouTube success?</h3>
        <p>Login or register to get started!</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔐 Login", use_container_width=True, type="primary", key="landing_login_button"):
            st.session_state.current_page = 'login'
            st.rerun()
    
    with col2:
        if st.button("📝 Register", use_container_width=True, key="landing_register_button"):
            st.session_state.current_page = 'register'
            st.rerun()

def login_page():
    st.header("**🔐 Login**")
    
    users = load_users()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login", type="primary", key="login_submit_button"):
            if username in users:
                if verify_password(password, users[username]["password"]):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.current_page = 'main'
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
            else:
                st.error("Invalid username or password")
        
        if st.button("Back to Landing", key="login_back_button"):
            st.session_state.current_page = 'landing'
            st.rerun()
    
    with col2:
        st.markdown("""
        ### Demo Account:
        - **Username:** admin
        - **Password:** admin123
        
        ### Features:
        - Content Analysis
        - Thumbnail Generation
        - URL Analysis
        - AI Recommendations
        """)

def register_page():
    st.header("**📝 Register**")
    
    users = load_users()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        new_username = st.text_input("New Username", key="register_username")
        new_password = st.text_input("New Password", type="password", key="register_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="register_confirm_password")
        
        if st.button("Register", type="primary", key="register_submit_button"):
            if new_username and new_password:
                if new_username in users:
                    st.error("Username already exists")
                elif new_password != confirm_password:
                    st.error("Passwords don't match")
                else:
                    # Hash password and save
                    hashed_password = hash_password(new_password)
                    users[new_username] = {
                        "name": new_username,
                        "password": hashed_password
                    }
                    save_users(users)
                    st.success("Registration successful! Please login.")
                    st.session_state.current_page = 'login'
                    st.rerun()
            else:
                st.error("Please fill all fields")
        
        if st.button("Back to Landing", key="register_back_button"):
            st.session_state.current_page = 'landing'
            st.rerun()
    
    with col2:
        st.markdown("""
        ### Registration Requirements:
        - Username must be unique
        - Password must be secure
        - All fields are required
        
        ### After Registration:
        - You'll be redirected to login
        - Use your new credentials
        - Access all features
        """)

def logout_page():
    st.header("**🚪 Logout**")
    
    st.markdown("Are you sure you want to logout?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Yes, Logout", type="primary", key="logout_confirm_button"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.current_page = 'landing'
            st.success("Logged out successfully!")
            st.rerun()
    
    with col2:
        if st.button("Cancel", key="logout_cancel_button"):
            st.session_state.current_page = 'main'
            st.rerun()

# --- Gemini Content Expert Chatbot ---
def content_creator_chatbot():
    st.header("**🤖 Content Creator Assistant**")

    # Configure model and safety once for single-shot advice
    genai.configure(api_key="AIzaSyCiWJwcl8u0tyOLYucXnA6JWlUwMMgvqbs")
    generation_config = {
        "temperature": 0.7,
        "top_p": 1,
        "top_k": 32,
        "max_output_tokens": 2000,
    }
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]
    model = get_gemini_model(
        prefer_multimodal=False,
        generation_config=generation_config,
        safety_settings=safety_settings,
    )

    expert_prompt = (
        "You are ContentGPT, an expert digital content creator with 10+ years of experience. "
        "You specialize in YouTube content strategy, scripting, SEO optimization, and audience engagement. "
        "Provide concise, actionable advice."
    )

    query = st.text_area(
        label="",
        height=100,
        key="assistant_query_single",
        placeholder="Type your content question here...",
        label_visibility="collapsed",
    )

    if st.button("Get Expert Advice", type="primary", key="assistant_get_advice_single"):
        with st.spinner("Generating advice..."):
            try:
                prompt = f"{expert_prompt}\n\nUser: {query}"
                response = safe_generate(model, prompt, prefer_multimodal=False)
                st.success("Advice:")
                st.markdown(response.text or "No response.")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Gemini model helper and initialization
def get_gemini_model(prefer_multimodal=False, generation_config=None, safety_settings=None):
    """Return a supported Gemini model instance using dynamic discovery only.
    Always pass the exact model name returned by the API to avoid 404s.
    """
    discovered = []  # exact names as returned by API, e.g. 'models/gemini-1.5-flash'
    try:
        models = genai.list_models()
        for m in models:
            try:
                methods = getattr(m, "supported_generation_methods", []) or []
                if any(str(x).lower() == "generatecontent" for x in methods):
                    name_full = getattr(m, "name", "") or ""
                    if name_full:
                        discovered.append(name_full)
            except Exception:
                continue
    except Exception:
        discovered = []

    # Helper to score and pick best model from discovered list
    def pick(discovered_names, prefer_mm=False):
        if not discovered_names:
            return None
        # Preferred substrings in order
        prefs_mm = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro-vision"]
        prefs_txt = ["gemini-1.5-pro", "gemini-1.5-flash"]
        prefs = prefs_mm if prefer_mm else prefs_txt
        # First pass: exact contains preferred tokens
        for token in prefs:
            for n in discovered_names:
                if token in n:
                    return n
        # Fallback: any with 'gemini' and '1.5'
        for n in discovered_names:
            if "gemini" in n and "1.5" in n:
                return n
        # Last fallback: any model supporting generateContent
        return discovered_names[0]

    chosen = pick(discovered, prefer_mm=prefer_multimodal)
    if chosen:
        try:
            if generation_config or safety_settings:
                return genai.GenerativeModel(
                    model_name=chosen,
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                )
            return genai.GenerativeModel(model_name=chosen)
        except Exception:
            pass

    # If discovery failed or chosen failed to init, try minimal hardcoded safe bets
    hardcoded = ["models/gemini-1.5-flash", "models/gemini-1.5-pro", "models/gemini-pro-vision"] if prefer_multimodal else ["models/gemini-1.5-pro", "models/gemini-1.5-flash"]
    for name in hardcoded:
        try:
            if generation_config or safety_settings:
                return genai.GenerativeModel(
                    model_name=name,
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                )
            return genai.GenerativeModel(model_name=name)
        except Exception:
            continue
    # As a last last resort, try bare names
    for name in ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro-vision"]:
        try:
            return genai.GenerativeModel(model_name=name)
        except Exception:
            continue
    raise RuntimeError("No available Gemini model found for generateContent")

# Safe generation helper to transparently retry with a fallback model
def safe_generate(model, content, prefer_multimodal=False, **kwargs):
    """Call model.generate_content with automatic fallback if the selected
    model isn't available for the current API version/region.
    """
    try:
        return model.generate_content(content, **kwargs)
    except Exception:
        # Try a fresh discovered-supported model and retry once
        try:
            fallback = get_gemini_model(prefer_multimodal=prefer_multimodal)
            return fallback.generate_content(content, **kwargs)
        except Exception:
            # As a final attempt, iterate discovered supported models directly
            try:
                models = genai.list_models()
                supported = []
                for m in models:
                    methods = getattr(m, "supported_generation_methods", []) or []
                    if any(str(x).lower() == "generatecontent" for x in methods):
                        name_full = getattr(m, "name", "") or ""
                        if name_full:
                            supported.append(name_full)
                # Prioritize by simple preference tokens
                tokens = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro-vision"] if prefer_multimodal else ["gemini-1.5-pro", "gemini-1.5-flash"]
                ordered = [n for t in tokens for n in supported if t in n]
                # Append any remaining supported
                ordered += [n for n in supported if n not in ordered]
                last_error = None
                for name in ordered:
                    try:
                        mdl = genai.GenerativeModel(model_name=name)
                        return mdl.generate_content(content, **kwargs)
                    except Exception as e3:
                        last_error = e3
                        continue
                raise last_error if last_error else RuntimeError("No discovered model succeeded")
            except Exception as e4:
                raise e4

# Initialize Gemini (call this once in your app)
def init_gemini():
    genai.configure(api_key="AIzaSyCiWJwcl8u0tyOLYucXnA6JWlUwMMgvqbs")
    return get_gemini_model(prefer_multimodal=True)

# Thumbnail Generator Tab
def thumbnail_tab(model):
    st.header("**🎨 Professional Thumbnail Studio**")
    
    with st.expander("⚙ Thumbnail Brief", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            video_title = st.text_input("Video Title*", placeholder="5 Secrets to Viral Videos", key="thumbnail_video_title")
            content_type = st.selectbox(
                "Content Type*",
                ["Tutorial", "Review", "Vlog", "Gaming", "Tech", "Fashion", "Other"],
                index=2,
                key="thumbnail_content_type"
            )
        with col2:
            target_audience = st.text_input("Target Audience*", placeholder="Ages 18-35, Tech Enthusiasts", key="thumbnail_target_audience")
            style_preference = st.multiselect(
                "Style Preferences",
                ["Minimalist", "Bold Text", "Face Close-up", "Product Focus", "Dark Theme", "Bright Colors"],
                default=["Bold Text"],
                key="thumbnail_style_preference"
            )
    
    with st.expander("🖼 Reference Image (Optional)"):
        uploaded_image = st.file_uploader("Upload image for inspiration", type=["jpg", "png", "jpeg"], key="thumbnail_reference_image")
        if uploaded_image:
            st.image(uploaded_image, caption="Your Reference", width=300)

    if st.button("✨ Generate Thumbnail Concepts", type="primary", key="generate_thumbnail_button"):
        if not video_title or not content_type or not target_audience:
            st.warning("Please fill required fields (*)")
            return
            
        with st.spinner("Generating 3 professional concepts..."):
            try:
                prompt = f"""As a YouTube thumbnail expert, create 3 distinct concepts for:
Title: {video_title}
Content: {content_type}
Audience: {target_audience}
Styles: {', '.join(style_preference) if style_preference else 'Any'}

For each concept provide:
1. 🎭 Visual Composition - Describe layout, colors, key elements
2. ✏ Text Treatment - Recommended text/fonts/placement
3. 🎯 Psychological Hook - Why it grabs attention
4. 💡 Pro Tip - Technical execution advice

Format with clear headings for each concept."""
                
                if uploaded_image:
                    img = Image.open(uploaded_image)
                    img_bytes = io.BytesIO()
                    img.save(img_bytes, format="PNG")
                    # Attempt multimodal generation; if unavailable, fall back to text-only
                    try:
                        response = safe_generate(
                            model,
                            [prompt, {"mime_type": "image/png", "data": base64.b64encode(img_bytes.getvalue()).decode()}],
                            prefer_multimodal=True,
                        )
                    except Exception:
                        text_only_model = get_gemini_model(prefer_multimodal=False)
                        response = safe_generate(text_only_model, prompt, prefer_multimodal=False)
                else:
                    # Use a safe text model to avoid access issues
                    text_model = get_gemini_model(prefer_multimodal=False)
                    response = safe_generate(text_model, prompt, prefer_multimodal=False)
                
                st.success("✅ Generated 3 Professional Concepts")
                st.session_state.last_thumbnail_response = response.text
                
                # Display with nice formatting
                for i, concept in enumerate(response.text.split("Concept ")[1:4], 1):
                    with st.container(border=True):
                        st.subheader(f"**Concept {i}**")
                        st.markdown(concept)
                
                # PDF download for thumbnail concepts
                try:
                    pdf_buf = io.BytesIO()
                    c = canvas.Canvas(pdf_buf, pagesize=letter)
                    width, height = letter
                    margin = 40
                    y = height - margin
                    def write_line(text, font="Helvetica", size=10, leading=14):
                        nonlocal y
                        c.setFont(font, size)
                        max_chars = 95
                        lines = [text[i:i+max_chars] for i in range(0, len(text), max_chars)] if text else [""]
                        for ln in lines:
                            c.drawString(margin, y, ln)
                            y -= leading
                            if y < margin:
                                c.showPage()
                                y = height - margin
                    c.setFont("Helvetica-Bold", 16)
                    c.drawString(margin, y, "YT Brain — Thumbnail Concepts Report")
                    y -= 24
                    write_line(f"Title: {video_title}")
                    write_line(f"Content Type: {content_type}")
                    write_line(f"Target Audience: {target_audience}")
                    write_line(f"Styles: {', '.join(style_preference) if style_preference else 'Any'}")
                    y -= 12
                    c.setFont("Helvetica-Bold", 12)
                    c.drawString(margin, y, "Generated Concepts")
                    y -= 16
                    c.setFont("Helvetica", 10)
                    for i, concept in enumerate(response.text.split("Concept ")[1:4], 1):
                        write_line(f"Concept {i}")
                        for paragraph in concept.split('\n'):
                            if paragraph.strip() == "":
                                y -= 6
                                continue
                            write_line(paragraph)
                        y -= 8
                    # attach uploaded reference image if provided
                    if uploaded_image:
                        try:
                            img = Image.open(uploaded_image)
                            img_buf = io.BytesIO()
                            img.save(img_buf, format='PNG')
                            img_reader = ImageReader(io.BytesIO(img_buf.getvalue()))
                            iw, ih = img_reader.getSize()
                            max_w = width - 2 * margin
                            scale = min(max_w / iw, (height - 2 * margin) / ih)
                            dw, dh = iw * scale, ih * scale
                            if y - dh < margin:
                                c.showPage()
                                y = height - margin
                            c.drawImage(img_reader, margin, y - dh, width=dw, height=dh, preserveAspectRatio=True, anchor='sw')
                            y -= (dh + 14)
                        except Exception:
                            pass
                    c.showPage()
                    c.save()
                    pdf_bytes = pdf_buf.getvalue()
                    pdf_buf.close()
                    st.download_button(
                        label="⬇️ Download PDF Report",
                        data=pdf_bytes,
                        file_name="thumbnail_concepts_report.pdf",
                        mime="application/pdf",
                        key="thumbnail_pdf_download"
                    )
                except Exception:
                    pass
                
            except Exception as e:
                st.error(f"🚨 Generation failed: {str(e)}")

def generate_ai_recommendations(topic, analysis_data=None):
    """Generate AI-powered content recommendations with enhanced strategy
    
    Args:
        topic (str): The main topic or niche to generate recommendations for
        analysis_data (dict, optional): Dictionary containing analytics data like:
            - avg_duration: Average video duration in minutes
            - top_keywords: List of top keywords
            - engagement_rate: Average engagement rate
            - best_performing_videos: List of best performing videos
    
    Returns:
        str: Formatted recommendations or error message
    """
    try:
        # Initialize a supported model using the helper (text-focused)
        model = get_gemini_model(prefer_multimodal=False)
        
        # Prepare analysis data for the prompt
        analysis_text = ""
        if analysis_data:
            analysis_text = "Analysis of your channel/videos shows:\n"
            if 'avg_duration' in analysis_data:
                analysis_text += f"- Average Video Duration: {analysis_data['avg_duration']:.1f} minutes\n"
            if 'top_keywords' in analysis_data:
                analysis_text += f"- Top Keywords: {', '.join(analysis_data['top_keywords'][:5])}\n"
            if 'engagement_rate' in analysis_data:
                analysis_text += f"- Engagement Rate: {analysis_data['engagement_rate']:.2%}\n"
        prompt = f"""As a YouTube strategy expert, provide comprehensive analysis and actionable recommendations for the topic: {topic}

{analysis_text}

Please provide detailed, specific recommendations including:

1. CONTENT STRATEGY:
   - 5 high-potential video ideas with unique angles
   - Content series or themes that would perform well
   - Types of content that generate most engagement

2. VIDEO OPTIMIZATION:
   - Ideal video length based on content type
   - Best days/times to post for maximum reach
   - Title and description optimization tips
   - Thumbnail design recommendations

3. AUDIENCE ENGAGEMENT:
   - Call-to-action strategies
   - Community building techniques
   - Ways to encourage likes, comments, and shares
   - Responding to audience feedback

4. DISCOVERABILITY:
   - Primary and secondary keywords to target
   - Hashtag strategy (10-15 relevant hashtags)
   - Playlist optimization
   - Cross-promotion opportunities

5. COMPETITIVE ANALYSIS:
   - What's working for top creators in this niche
   - Gaps in the market to exploit
   - Unique value proposition development

6. TRENDING OPPORTUNITIES:
   - Current trending topics in this niche
   - Seasonal content opportunities
   - Emerging trends to capitalize on

Make the recommendations specific, actionable, and tailored to help grow the channel. Focus on providing concrete, practical advice that can be implemented immediately."""
        
        # Configure generation settings
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
        
        # Set safety settings
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        
        # Generate the response
        response = safe_generate(
            model,
            prompt,
            prefer_multimodal=False,
            generation_config=generation_config,
            safety_settings=safety_settings,
        )
        
        return response.text if response.text else "No response generated. Please try again with more specific details."
    
    except Exception as e:
        return f"AI recommendation error: {str(e)}"

def content_analysis_tab():
    """Content analysis functionality from app1.py"""
    st.header("**📊 Advanced YouTube Content Analysis**")
    st.markdown("Analyze YouTube content trends, get video suggestions, and discover what works best for your niche.")
    
    col1, col2 = st.columns(2)
    with col1:
        topic = st.text_input("Enter YouTube Topic", "Example - Deep Learning", key="content_analysis_topic")
        max_results = st.slider("Number of Videos to Analyze", 5, 50, 20, key="content_analysis_max_results")
    with col2:
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Most Views", "Most Comments", "Most Likes", "High Engagement", "Recent Videos", "Mixed Analysis"],
            key="content_analysis_type"
        )
        include_comments = st.checkbox("Include Comment Analysis", value=True, key="content_analysis_comments")

    if st.button("🚀 Analyze Content", type="primary", key="content_analysis_button"):
        with st.spinner("Analyzing YouTube content trends..."):
            try:
                # Use yt-dlp to search for videos instead of YouTube API
                st.info("🔍 Searching for videos using yt-dlp (no API quota required)...")
                
                # Create a simple search using yt-dlp
                search_query = topic
                analyzed_videos = []
                
                # Search for videos using yt-dlp with improved error handling
                ydl_opts = {
                    'quiet': True,
                    'no_warnings': True,
                    'extract_flat': True,
                    'default_search': 'ytsearch',
                    'max_downloads': max_results,
                    'ignoreerrors': True,
                    'skip_download': True,
                    'noplaylist': True,
                    'geo_bypass': True,
                    'http_headers': {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'
                    },
                    'extractor_args': {
                        'youtube': {
                            'player_client': ['android', 'web']
                        }
                    },
                }
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    try:
                        # Clean and improve search query
                        clean_query = search_query.strip()
                        if not clean_query:
                            clean_query = "popular videos"
                        
                        st.info(f"🔍 Searching for: '{clean_query}'...")
                        
                        # Search for videos
                        search_results = ydl.extract_info(f"ytsearch{max_results}:{clean_query}", download=False)
                        
                        if search_results and 'entries' in search_results and search_results['entries']:
                            valid_entries = [entry for entry in search_results['entries'] if entry is not None]
                            st.success(f"✅ Found {len(valid_entries)} videos to analyze")
                            
                            # Analyze each video with better error handling
                            for i, entry in enumerate(valid_entries):
                                try:
                                    # Use basic info from search first
                                    title = entry.get('title', f'Video {i+1}')
                                    video_id = entry.get('id', '')
                                    uploader = entry.get('uploader', 'Unknown Channel')
                                    duration = entry.get('duration', 0)
                                    view_count = entry.get('view_count', 0)
                                    
                                    # Try to get more detailed info if available
                                    video_url = f"https://www.youtube.com/watch?v={video_id}" if video_id else entry.get('url', '')
                                    
                                    if video_url:
                                        try:
                                            # Get detailed video info with simpler options
                                            detailed_opts = {
                                                'quiet': True,
                                                'no_warnings': True,
                                                'extract_flat': False,
                                                'ignoreerrors': True,
                                                'skip_download': True,
                                                'noplaylist': True,
                                                'geo_bypass': True,
                                                'http_headers': {
                                                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'
                                                },
                                                'extractor_args': {
                                                    'youtube': {
                                                        'player_client': ['android', 'web']
                                                    }
                                                },
                                            }
                                            
                                            with yt_dlp.YoutubeDL(detailed_opts) as detailed_ydl:
                                                detailed_info = detailed_ydl.extract_info(video_url, download=False)
                                                
                                                if detailed_info:
                                                    # Update with detailed info
                                                    title = detailed_info.get('title', title)
                                                    duration = detailed_info.get('duration', duration)
                                                    uploader = detailed_info.get('uploader', uploader)
                                                    view_count = detailed_info.get('view_count', view_count)
                                                    like_count = detailed_info.get('like_count', 0)
                                                    description = detailed_info.get('description', '')
                                                    upload_date = detailed_info.get('upload_date', '')
                                                    tags = detailed_info.get('tags', [])
                                                else:
                                                    # Use basic info if detailed extraction fails
                                                    like_count = 0
                                                    description = ''
                                                    upload_date = ''
                                                    tags = []
                                        except:
                                            # Fallback to basic info
                                            like_count = 0
                                            description = ''
                                            upload_date = ''
                                            tags = []
                                    else:
                                        like_count = 0
                                        description = ''
                                        upload_date = ''
                                        tags = []
                                    
                                    # Calculate engagement
                                    engagement = 0
                                    if view_count and like_count and view_count > 0:
                                        engagement = (like_count / view_count) * 100
                                    
                                    # Calculate video age
                                    video_age_days = 0
                                    if upload_date:
                                        try:
                                            upload_datetime = datetime.strptime(upload_date, '%Y%m%d')
                                            video_age_days = (datetime.now() - upload_datetime).days
                                        except:
                                            pass
                                    
                                    # Add to analyzed videos
                                    analyzed_videos.append({
                                        'title': title,
                                        'channel': uploader,
                                        'duration_mins': duration / 60 if duration else 0,
                                        'views': view_count if view_count else 0,
                                        'likes': like_count if like_count else 0,
                                        'url': video_url,
                                        'engagement': engagement,
                                        'description': description,
                                        'upload_date': upload_date,
                                        'video_age_days': video_age_days,
                                        'tags': tags
                                    })
                                    
                                    st.success(f"✅ Analyzed: {title[:50]}...")
                                    
                                except Exception as e:
                                    st.warning(f"⚠️ Could not analyze video {i+1}: {str(e)}")
                                    continue
                        else:
                            st.error("❌ No videos found for the search query. Try a different topic or check your internet connection.")
                            return
                            
                    except Exception as e:
                        st.error(f"❌ Search failed: {str(e)}")
                        st.info("💡 **Troubleshooting Tips:**\n- Try a simpler search term\n- Check your internet connection\n- Reduce the number of videos to analyze")
                        return
                
                if analyzed_videos:
                    st.success(f"✅ Successfully analyzed {len(analyzed_videos)} videos!")
                    
                    # Create dataframe
                    df = pd.DataFrame(analyzed_videos)
                    
                    # Display results
                    st.subheader("**📊 Analysis Results**")
                    
                    # Show summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        avg_duration = df['duration_mins'].mean() if not df.empty and df['duration_mins'].notna().any() else 0
                        st.metric("Avg Duration", f"{avg_duration:.1f} min")
                    with col2:
                        total_views = df['views'].sum() if not df.empty and df['views'].notna().any() else 0
                        st.metric("Total Views", f"{total_views:,}")
                    with col3:
                        avg_engagement = df['engagement'].mean() if not df.empty and df['engagement'].notna().any() else 0
                        st.metric("Avg Engagement", f"{avg_engagement:.2f}%")
                    with col4:
                        total_videos = len(df)
                        st.metric("Videos Analyzed", total_videos)
                    
                    # Enhanced video recommendations
                    st.subheader("**🎯 Smart Video Recommendations**")
                    
                    # Create different recommendation categories
                    recommendations = {}
                    
                    # High Engagement Videos
                    if not df.empty and 'engagement' in df.columns and df['engagement'].notna().any():
                        recommendations["🔥 High Engagement Videos"] = df.nlargest(5, 'engagement')
                    else:
                        recommendations["🔥 High Engagement Videos"] = pd.DataFrame()
                    
                    # High View Count
                    if not df.empty and 'views' in df.columns and df['views'].notna().any():
                        recommendations["📈 High View Count"] = df.nlargest(5, 'views')
                    else:
                        recommendations["📈 High View Count"] = pd.DataFrame()
                    
                    # Optimal Duration
                    duration_filter = df[(df['duration_mins'] >= 5) & (df['duration_mins'] <= 15)]
                    if not duration_filter.empty and 'engagement' in duration_filter.columns and duration_filter['engagement'].notna().any():
                        recommendations["⏱️ Optimal Duration (5-15 min)"] = duration_filter.nlargest(5, 'engagement')
                    else:
                        recommendations["⏱️ Optimal Duration (5-15 min)"] = pd.DataFrame()
                    
                    # Trending Topics
                    if not df.empty and 'views' in df.columns and df['views'].notna().any():
                        recommendations["💡 Trending Topics"] = df.nlargest(5, 'views')
                    else:
                        recommendations["💡 Trending Topics"] = pd.DataFrame()
                    
                    # Display recommendations in tabs
                    rec_tabs = st.tabs(list(recommendations.keys()))
                    
                    for i, (category, rec_df) in enumerate(recommendations.items()):
                        with rec_tabs[i]:
                            if not rec_df.empty:
                                for idx, row in rec_df.iterrows():
                                    with st.container(border=True):
                                        col1, col2, col3 = st.columns([3, 1, 1])
                                        with col1:
                                            st.markdown(f"**{row['title']}**")
                                            st.caption(f"Channel: {row['channel']}")
                                            st.markdown(f"🔗 [Watch on YouTube]({row['url']})")
                                        with col2:
                                            st.metric("Duration", f"{row['duration_mins']:.1f} min")
                                            st.metric("Views", f"{row['views']:,}")
                                        with col3:
                                            st.metric("Engagement", f"{row['engagement']:.2f}%")
                                            if row['video_age_days'] > 0:
                                                st.metric("Age", f"{row['video_age_days']} days")
                            else:
                                st.info(f"No videos found for {category}")
                    
                    
                    # Content analysis with improved insights
                    st.subheader("**🧠 Advanced Content Analysis**")
                    
                    # Analyze titles for content types
                    if not df.empty and 'title' in df.columns:
                        all_titles = ' '.join(df['title'].str.lower())
                    else:
                        all_titles = ''
                    
                    if not df.empty and 'description' in df.columns:
                        all_descriptions = ' '.join([str(desc) for desc in df['description'] if pd.notna(desc)])
                    else:
                        all_descriptions = ''
                    
                    # Enhanced content type detection
                    content_types = {
                        'music': ['music', 'song', 'audio', 'instrumental', 'melody', 'beats', 'soundtrack', 'cover'],
                        'gaming': ['game', 'gaming', 'playthrough', 'stream', 'gamer', 'gameplay', 'esports'],
                        'tutorial': ['tutorial', 'how to', 'guide', 'learn', 'education', 'tips', 'tricks'],
                        'vlog': ['vlog', 'daily', 'life', 'day in the life', 'lifestyle', 'personal'],
                        'review': ['review', 'test', 'comparison', 'analysis', 'unboxing', 'product'],
                        'tech': ['technology', 'tech', 'software', 'programming', 'coding', 'ai', 'machine learning'],
                        'fitness': ['workout', 'fitness', 'exercise', 'training', 'gym', 'health'],
                        'cooking': ['recipe', 'cooking', 'food', 'kitchen', 'chef', 'baking']
                    }
                    
                    content_scores = {}
                    for content_type, keywords in content_types.items():
                        score = sum(1 for word in keywords if word in all_titles or word in all_descriptions)
                        content_scores[content_type] = score
                    
                    
                    # Content Type Distribution with URLs
                    st.subheader("**📊 Content Type Distribution with URLs**")
                    
                    # Categorize videos by content type
                    video_categories = {}
                    for content_type in content_types.keys():
                        video_categories[content_type] = []
                    
                    # Analyze each video and categorize it
                    for _, video in df.iterrows():
                        video_title = str(video.get('title', '')).lower()
                        video_desc = str(video.get('description', '')).lower()
                        video_matched = False
                        
                        # Check which content type this video belongs to
                        for content_type, keywords in content_types.items():
                            if any(keyword in video_title or keyword in video_desc for keyword in keywords):
                                video_categories[content_type].append({
                                    'title': video.get('title', 'Unknown'),
                                    'url': video.get('url', ''),
                                    'channel': video.get('channel', 'Unknown'),
                                    'views': video.get('views', 0),
                                    'engagement': video.get('engagement', 0)
                                })
                                video_matched = True
                                break
                        
                        # If no category matched, add to 'other'
                        if not video_matched:
                            if 'other' not in video_categories:
                                video_categories['other'] = []
                            video_categories['other'].append({
                                'title': video.get('title', 'Unknown'),
                                'url': video.get('url', ''),
                                'channel': video.get('channel', 'Unknown'),
                                'views': video.get('views', 0),
                                'engagement': video.get('engagement', 0)
                            })
                    
                    # Display categorized videos
                    for content_type, videos in video_categories.items():
                        if videos:  # Only show categories that have videos
                            with st.expander(f"**{content_type.title()} Content ({len(videos)} videos)**", expanded=False):
                                for i, video in enumerate(videos, 1):
                                    col1, col2, col3 = st.columns([3, 1, 1])
                                    with col1:
                                        st.markdown(f"**{i}. {video['title'][:60]}{'...' if len(video['title']) > 60 else ''}**")
                                        st.caption(f"Channel: {video['channel']}")
                                        if video['url']:
                                            st.markdown(f"🔗 [Watch Video]({video['url']})")
                                    with col2:
                                        st.metric("Views", f"{video['views']:,}")
                                    with col3:
                                        st.metric("Engagement", f"{video['engagement']:.2f}%")
                                    st.divider()
                    
                    # Performance insights with enhanced metrics
                    st.subheader("**📈 Performance Insights**")
                    
                    if not df.empty:
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            avg_engagement = df['engagement'].mean() if not df.empty and df['engagement'].notna().any() else 0
                            st.metric("Average Engagement", f"{avg_engagement:.3f}%")
                        with col2:
                            max_engagement = df['engagement'].max() if not df['engagement'].empty and df['engagement'].notna().any() else 0
                            st.metric("Highest Engagement", f"{max_engagement:.3f}%")
                        with col3:
                            avg_views = df['views'].mean() if not df.empty and df['views'].notna().any() else 0
                            st.metric("Average Views", f"{avg_views:,.0f}")
                        with col4:
                            total_videos = len(df)
                            st.metric("Videos Analyzed", total_videos)
                    
                    # Enhanced AI Recommendations
                    st.subheader("**🧠 AI-Powered Strategy Recommendations**")
                    rec_text = None
                    
                    # Prepare enhanced analysis data
                    analysis_data = {
                        'avg_duration': df['duration_mins'].mean() if not df.empty and df['duration_mins'].notna().any() else 0,
                        'best_hours': [],  # Not available without API
                        'top_keywords': list(content_scores.keys()),
                        'sentiment': 0,  # Not available without API
                        'avg_engagement': df['engagement'].mean() if not df.empty and df['engagement'].notna().any() else 0,
                        'avg_views': df['views'].mean() if not df.empty and df['views'].notna().any() else 0,
                        'content_distribution': content_scores
                    }
                    
                    # Add basic topic overview
                    st.markdown(f"""
                    ### **📋 Topic Overview: {topic}**
                    
                    **What is this topic about?**
                    - {topic} is a popular content category on YouTube
                    - Average engagement rate: {analysis_data['avg_engagement']:.2f}%
                    - Typical video length: {analysis_data['avg_duration']:.1f} minutes
                    
                    **Why this topic works on YouTube:**
                    - High search volume and viewer interest
                    - Diverse content formats (tutorials, reviews, entertainment)
                    - Strong community engagement
                    - Monetization opportunities through sponsorships and ads
                    """)
                    
                    try:
                        model = init_gemini()
                        enhanced_prompt = f"""As a YouTube strategy expert specializing in {topic}, analyze this data and provide comprehensive, topic-specific recommendations:
                        
                        Topic: {topic}
                        Analysis Type: {analysis_type}
                        Videos Analyzed: {len(df)}
                        
                        Performance Analysis:
                        - Average Duration: {analysis_data['avg_duration']:.1f} minutes
                        - Average Engagement: {analysis_data['avg_engagement']:.2f}%
                        - Average Views: {analysis_data['avg_views']:,.0f}
                        - Content Types: {list(content_scores.keys())}
                        
                        Provide topic-specific recommendations for "{topic}":
                        
                        1. **5 Viral Title Ideas** - Create catchy, topic-specific titles that would work for "{topic}"
                        2. **Ideal Video Length** - What's the perfect duration for "{topic}" content based on audience behavior
                        3. **Content Strategy Tips** - Specific strategies for "{topic}" creators based on top performers
                        4. **10 Trending Hashtags** - Relevant hashtags specifically for "{topic}" content
                        5. **3 Specific Video Ideas** - Concrete video concepts for "{topic}" that would perform well
                        6. **Engagement Optimization** - How to increase engagement specifically for "{topic}" videos
                        7. **Best Posting Schedule** - When to post "{topic}" content for maximum reach
                        8. **Topic-Specific Insights** - What makes "{topic}" content successful vs other niches
                        9. **Audience Targeting** - Who watches "{topic}" content and how to reach them
                        10. **Monetization Tips** - How to monetize "{topic}" content effectively
                        
                        Make all recommendations highly specific to "{topic}" and actionable for creators in this niche."""
                        
                        recommendations = safe_generate(model, enhanced_prompt, prefer_multimodal=False)
                        rec_text = recommendations.text
                        st.markdown(rec_text)
                    except Exception as e:
                        st.warning(f"⚠️ AI recommendations failed: {str(e)}")
                        st.info("""
                        **Basic Strategy Recommendations:**
                        - Focus on high-engagement content types
                        - Optimize video length based on audience preferences
                        - Use trending keywords and hashtags
                        - Post consistently and engage with audience
                        - Analyze competitor content for inspiration
                        """)
                    
                    # Enhanced Visualization Section
                    st.subheader("**📊 Advanced Data Visualizations**")
                    images_for_pdf = []
                    
                    try:
                        # Clean data before visualization to handle NaN values
                        df_clean = df.copy()
                        
                        # Replace NaN values with 0 for numeric columns
                        numeric_columns = ['duration_mins', 'views', 'likes', 'engagement', 'video_age_days']
                        for col in numeric_columns:
                            if col in df_clean.columns:
                                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
                        
                        # Remove rows where all key metrics are 0 or NaN
                        df_clean = df_clean[(df_clean['views'] > 0) | (df_clean['duration_mins'] > 0)]
                        
                        if df_clean.empty:
                            st.info("📊 No valid data available for visualization after cleaning")
                        else:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**📈 Video Duration Distribution**")
                                if 'duration_mins' in df_clean.columns and df_clean['duration_mins'].sum() > 0:
                                    # Filter out zero durations for better visualization
                                    duration_data = df_clean[df_clean['duration_mins'] > 0]['duration_mins']
                                    if len(duration_data) > 0:
                                        fig, ax = plt.subplots(figsize=(8, 6))
                                        sns.histplot(duration_data, bins=min(10, len(duration_data)), kde=True, ax=ax)
                                        plt.xlabel("Duration (minutes)")
                                        plt.ylabel("Number of Videos")
                                        plt.title("Video Duration Distribution")
                                        plt.tight_layout()
                                        st.pyplot(fig)
                                        try:
                                            buf = io.BytesIO()
                                            fig.savefig(buf, format='png', bbox_inches='tight')
                                            images_for_pdf.append(buf.getvalue())
                                            buf.close()
                                        except Exception:
                                            pass
                                        plt.close()
                                    else:
                                        st.info("📊 No valid duration data for visualization")
                                else:
                                    st.info("📊 Duration data not available for visualization")
                            
                            with col2:
                                st.markdown("**📊 Engagement Analysis**")
                                if 'engagement' in df_clean.columns and df_clean['engagement'].sum() > 0:
                                    # Filter out zero engagement for better visualization
                                    engagement_data = df_clean[df_clean['engagement'] > 0]['engagement']
                                    if len(engagement_data) > 0:
                                        fig, ax = plt.subplots(figsize=(8, 6))
                                        # Create engagement categories with proper handling
                                        engagement_categories = pd.cut(engagement_data, 
                                                                     bins=[0, 1, 5, 10, float('inf')], 
                                                                     labels=['Low (0-1%)', 'Medium (1-5%)', 'High (5-10%)', 'Very High (10%+)'],
                                                                     include_lowest=True)
                                        
                                        engagement_counts = engagement_categories.value_counts()
                                        if len(engagement_counts) > 0:
                                            colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4'][:len(engagement_counts)]
                                            
                                            plt.pie(engagement_counts.values, labels=engagement_counts.index, 
                                                   autopct='%1.1f%%', colors=colors, startangle=90)
                                            plt.title("Engagement Level Distribution")
                                            plt.axis('equal')
                                            st.pyplot(fig)
                                            try:
                                                buf = io.BytesIO()
                                                fig.savefig(buf, format='png', bbox_inches='tight')
                                                images_for_pdf.append(buf.getvalue())
                                                buf.close()
                                            except Exception:
                                                pass
                                            plt.close()
                                        else:
                                            st.info("📊 No engagement categories to display")
                                    else:
                                        st.info("📊 No valid engagement data for visualization")
                                else:
                                    st.info("📊 Engagement data not available for visualization")
                            
                            # Additional visualizations
                            col3, col4 = st.columns(2)
                            
                            with col3:
                                st.markdown("**📺 Views vs Engagement**")
                                if 'views' in df_clean.columns and 'engagement' in df_clean.columns:
                                    # Filter data for scatter plot
                                    scatter_data = df_clean[(df_clean['views'] > 0) & (df_clean['engagement'] > 0)]
                                    if len(scatter_data) > 0:
                                        fig, ax = plt.subplots(figsize=(8, 6))
                                        plt.scatter(scatter_data['views'], scatter_data['engagement'], alpha=0.7, s=50, color='#4ecdc4')
                                        plt.xlabel("Views")
                                        plt.ylabel("Engagement Rate (%)")
                                        plt.title("Views vs Engagement Rate")
                                        if scatter_data['views'].max() > 1000:
                                            plt.xscale('log')  # Log scale for views only if needed
                                        plt.grid(True, alpha=0.3)
                                        st.pyplot(fig)
                                        try:
                                            buf = io.BytesIO()
                                            fig.savefig(buf, format='png', bbox_inches='tight')
                                            images_for_pdf.append(buf.getvalue())
                                            buf.close()
                                        except Exception:
                                            pass
                                        plt.close()
                                    else:
                                        st.info("📊 No valid data points for scatter plot")
                                else:
                                    st.info("📊 Views/Engagement data not available")
                            
                            with col4:
                                st.markdown("**🎯 Top Performing Videos**")
                                if 'engagement' in df_clean.columns and len(df_clean) > 0:
                                    # Get top videos by engagement (filter out zero engagement)
                                    top_videos = df_clean[df_clean['engagement'] > 0].nlargest(min(5, len(df_clean)), 'engagement')
                                    
                                    if not top_videos.empty and len(top_videos) > 0:
                                        fig, ax = plt.subplots(figsize=(8, 6))
                                        y_pos = range(len(top_videos))
                                        bars = plt.barh(y_pos, top_videos['engagement'], color='#4ecdc4')
                                        
                                        # Create safe labels
                                        labels = []
                                        for title in top_videos['title']:
                                            if pd.isna(title) or title == '':
                                                labels.append('Unknown Video')
                                            else:
                                                safe_title = str(title)[:30]
                                                labels.append(safe_title + '...' if len(str(title)) > 30 else safe_title)
                                        
                                        plt.yticks(y_pos, labels)
                                        plt.xlabel("Engagement Rate (%)")
                                        plt.title("Top Performing Videos")
                                        
                                        # Add value labels on bars
                                        for i, (bar, engagement) in enumerate(zip(bars, top_videos['engagement'])):
                                            width = bar.get_width()
                                            if not pd.isna(width) and width > 0:
                                                plt.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                                                        f'{width:.2f}%', ha='left', va='center')
                                        
                                        plt.tight_layout()
                                        st.pyplot(fig)
                                        try:
                                            buf = io.BytesIO()
                                            fig.savefig(buf, format='png', bbox_inches='tight')
                                            images_for_pdf.append(buf.getvalue())
                                            buf.close()
                                        except Exception:
                                            pass
                                        plt.close()
                                    else:
                                        st.info("📊 No videos with engagement data available")
                                else:
                                    st.info("📊 Engagement data not available for visualization")
                            
                    except Exception as e:
                        st.warning(f"⚠️ Visualization failed: cannot convert float NaN to integer")
                        st.info("""
                        💡 **Alternative:**
                        - Check the data table above for detailed metrics
                        - Use the recommendations section for insights
                        """)
                    
                    # Show full results table
                    st.subheader("**📋 Complete Analysis Data**")
                    if not df.empty:
                        # Check which columns exist before displaying
                        available_columns = ['title', 'channel', 'duration_mins', 'views', 'engagement', 'video_age_days']
                        existing_columns = [col for col in available_columns if col in df.columns]
                        
                        if existing_columns:
                            display_df = df[existing_columns].copy()
                            st.dataframe(display_df, use_container_width=True)
                            # PDF export
                            try:
                                summary_metrics = {
                                    "Average Duration": f"{df['duration_mins'].mean():.1f} min" if 'duration_mins' in df.columns else "N/A",
                                    "Total Views": f"{int(df['views'].sum()):,}" if 'views' in df.columns else "0",
                                    "Average Engagement": f"{df['engagement'].mean():.2f}%" if 'engagement' in df.columns else "0%",
                                    "Videos Analyzed": f"{len(df)}"
                                }
                                pdf_bytes = generate_content_pdf(topic, analysis_type, df, summary_metrics, rec_text=rec_text, images=images_for_pdf)
                                st.download_button(
                                    label="⬇️ Download PDF Report",
                                    data=pdf_bytes,
                                    file_name=f"yt_content_analysis_{re.sub(r'[^a-zA-Z0-9]+', '_', topic)[:40]}.pdf",
                                    mime="application/pdf",
                                    key="content_analysis_pdf_download"
                                )
                            except Exception as e:
                                st.warning(f"PDF export unavailable: {str(e)}")
                        else:
                            st.info("📊 No data available for display")
                    else:
                        st.info("📊 No data available for display")
                    
                else:
                    st.error("❌ No videos could be analyzed. Please try a different topic.")
                
            except Exception as e:
                st.error(f"""
                ❌ **Analysis Failed**
                
                Error: {str(e)}
                
                **Possible causes:**
                - Network connection issues
                - Invalid search query
                - YouTube access problems
                
                **Solutions:**
                1. Check your internet connection
                2. Try a different topic
                3. Reduce the number of videos to analyze
                """)

def main_app():
    st.title("🎥 YT Brain: The AI Toolkit for YouTube Success")
    
    # User info and logout
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"Welcome back, **{st.session_state.username}**! 👋")
    with col2:
        if st.button("🚪 Logout", key="main_logout_button"):
            st.session_state.current_page = 'logout'
            st.rerun()
    
    # Initialize AI models
    model = init_gemini()
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 URL Analysis", "📊 Content Analysis", "🤖 Content Assistant", "🎨 Thumbnail Studio"])
    
    with tab1:
        audio_download_page()
    
    with tab2:
        content_analysis_tab()
    
    with tab3:
        content_creator_chatbot()
    
    with tab4:
        thumbnail_tab(model)

def main():
    # Page routing
    if st.session_state.current_page == 'landing':
        landing_page()
    elif st.session_state.current_page == 'login':
        login_page()
    elif st.session_state.current_page == 'register':
        register_page()
    elif st.session_state.current_page == 'logout':
        logout_page()
    elif st.session_state.current_page == 'main' and st.session_state.authenticated:
        main_app()
    else:
        st.session_state.current_page = 'landing'
        landing_page()

if __name__ == "__main__":
    main()