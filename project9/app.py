import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import datetime as dt
from tkinter import Tk, Label, Text, Frame, Canvas
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load browser history CSV (replace with your actual path)
data = pd.read_csv('browser_history.csv')

# Clean the data (remove irrelevant records, e.g., search queries, duplicate visits)
data.dropna(subset=['url', 'timestamp'], inplace=True)
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Extract domain and session features
data['domain'] = data['url'].str.extract(r'([a-zA-Z0-9-]+\.[a-zAZ]{2,})')
data['hour'] = data['timestamp'].dt.hour
data['day_of_week'] = data['timestamp'].dt.dayofweek

# Define categories for URLs (expand based on needs)
categories = {
    'health': ['healthline', 'fitness', 'wellness', 'medical'],
    'shopping': ['amazon', 'nike', 'ebay', 'shop'],
    'news': ['nytimes', 'bbc', 'cnn', 'theguardian'],
    'social_media': ['facebook', 'instagram', 'twitter', 'reddit'],
    'technology': ['techcrunch', 'wired', 'theverge'],
    'education': ['coursera', 'edx', 'khanacademy'],
}

def categorize_domain(domain):
    for category, keywords in categories.items():
        if any(keyword in domain for keyword in keywords):
            return category
    return 'other'

# Categorize domains
data['category'] = data['domain'].apply(categorize_domain)

# Feature Extraction: Frequency-based categorization
interest_summary = data['category'].value_counts()

# Activity Pattern Analysis (Session duration, Peak activity hours)
activity_by_hour = data.groupby('hour').size()
session_duration_by_category = data.groupby('category')['hour'].mean()

# KMeans Clustering to predict interests (Unsupervised)
# Use URL categories to create a clustering model of user interests
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['category'])
kmeans = KMeans(n_clusters=5, random_state=0)
data['cluster'] = kmeans.fit_predict(X)

# Demographics (Age and Location prediction based on URL analysis)
def infer_age_group(url):
    if 'instagram' in url or 'tiktok' in url:
        return '18-25'
    elif 'linkedin' in url or 'career' in url:
        return '25-40'
    else:
        return '40+'

data['age_group'] = data['url'].apply(infer_age_group)

# Location prediction based on domain geolocation (simplified)
def infer_location(domain):
    if 'uk' in domain:
        return 'United Kingdom'
    elif 'in' in domain:
        return 'India'
    else:
        return 'Other'

data['location'] = data['domain'].apply(infer_location)

# Risk Detection: Addictions (Frequent visits to gaming sites)
addictive_sites = ['gaming', 'gambling']
data['addiction_risk'] = data['url'].apply(lambda x: any(site in x for site in addictive_sites))

# Privacy Concerns: Suspected risky domains (based on blacklist)
risky_domains = ['phishing', 'malware', 'data-leak']
data['privacy_risk'] = data['url'].apply(lambda x: any(domain in x for domain in risky_domains))

# Final Insights (Concise output of predicted user profile)
def generate_profile(data):
    profile = {
        'Top Interests': data['category'].value_counts().idxmax(),
        'Predicted Age Group': data['age_group'].mode()[0],
        'Predicted Location': data['location'].mode()[0],
        'Addiction Risk': 'High' if data['addiction_risk'].sum() > len(data) * 0.05 else 'Low',
        'Privacy Risk': 'High' if data['privacy_risk'].sum() > len(data) * 0.05 else 'Low',
        'Frequent Categories': data['category'].mode()[0],
        'Peak Activity Hour': data['hour'].mode()[0],
        'Frequent Domain': data['domain'].mode()[0],
        'Shopping Preference': 'Yes' if 'amazon' in data['url'].mode()[0] else 'No',
        'Health Risk': 'High' if 'fitness' in data['category'].mode()[0] else 'Low',
        'Tech Affinity': 'Yes' if 'techcrunch' in data['url'].mode()[0] else 'No',
        'Social Media Usage': 'High' if 'instagram' in data['url'].mode()[0] else 'Low',
        'Education Affinity': 'Yes' if 'coursera' in data['url'].mode()[0] else 'No',
        'News Consumption': 'High' if 'bbc' in data['url'].mode()[0] else 'Low',
        'Frequent Content Type': data['category'].value_counts().idxmax(),
        'Preferred Shopping Sites': 'Amazon' if 'amazon' in data['url'].mode()[0] else 'Others',
        'Favorite Brands': 'Nike' if 'nike' in data['url'].mode()[0] else 'Others',
        'Time Spent on Social Media': 'High' if 'facebook' in data['url'].mode()[0] else 'Low',
        'Peak Time of Activity': 'Evening' if data['hour'].mode()[0] >= 18 else 'Morning',
        'Frequency of Visits': data.groupby('url').size().max(),
        'Personal Health Interest': 'High' if 'healthline' in data['url'].mode()[0] else 'Low',
        'E-Commerce Activity': 'High' if 'ebay' in data['url'].mode()[0] else 'Low',
        'Job-Related Interest': 'High' if 'linkedin' in data['url'].mode()[0] else 'Low',
        'Gaming Behavior': 'Frequent' if 'gaming' in data['url'].mode()[0] else 'Rare',
        'Music & Entertainment Preference': 'High' if 'spotify' in data['url'].mode()[0] else 'Low',
        'Political Interest': 'High' if 'nytimes' in data['url'].mode()[0] else 'Low',
        'Work-Related Focus': 'High' if 'microsoft' in data['url'].mode()[0] else 'Low',
        'Tech-Savvy': 'Yes' if 'theverge' in data['url'].mode()[0] else 'No',
        'Mobile Usage': 'High' if 'instagram' in data['url'].mode()[0] else 'Low',
        'Web Development Interest': 'Yes' if 'stackoverflow' in data['url'].mode()[0] else 'No'
    }
    return profile

user_profile = generate_profile(data)

# Create a Tkinter window with a scrollable frame to show the UI
def show_ui():
    window = Tk()
    window.title("User Profile Analysis")
    window.geometry("900x700")

    # Adding a title and company name at the top
    title_frame = Frame(window)
    title_frame.pack(fill="x", pady=10)
    
    title_label = Label(title_frame, text="User Profile Analysis by Browsing History", font=("Arial", 20, "bold"), fg="white")
    title_label.pack(side="left", padx=20)

    company_label = Label(title_frame, text="Powered by Browisify", font=("Arial", 12), fg="gray")
    company_label.pack(side="right", padx=20)

    # Create a style for the scrollbar
    style = ttk.Style()
    style.configure("TScrollbar", gripcount=0, background="darkblue", darkcolor="darkblue", lightcolor="blue")

    # Create a canvas for scrolling
    canvas = Canvas(window)
    scroll_y = ttk.Scrollbar(window, orient="vertical", command=canvas.yview, style="TScrollbar")
    frame = Frame(canvas)

    canvas.configure(yscrollcommand=scroll_y.set)
    scroll_y.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    canvas.create_window((0, 0), window=frame, anchor="nw")

    # Labels and Text output
    label = Label(frame, text="User Profile Summary", font=("Arial", 16))
    label.grid(row=0, column=0, padx=10, pady=10)

    text_output = Text(frame, height=25, width=100)
    for idx, (key, value) in enumerate(user_profile.items(), 1):
        text_output.insert(f"{idx}.0", f"{key}: {value}\n")

    text_output.grid(row=1, column=0, padx=10, pady=10)

    # Visualize Interests (Pie Chart)
    fig1, ax1 = plt.subplots(figsize=(5, 3))  # Reduced size
    ax1.pie(interest_summary, labels=interest_summary.index, autopct='%1.1f%%', colors=['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0'])
    ax1.set_title("User Interests Breakdown")

    canvas1 = FigureCanvasTkAgg(fig1, master=frame)
    canvas1.get_tk_widget().grid(row=2, column=0, padx=10, pady=10)
    description = "This chart shows the distribution of user's top interests based on their browsing behavior."

    label_desc1 = Label(frame, text=description, font=("Arial", 10), wraplength=700)
    label_desc1.grid(row=3, column=0, padx=10, pady=10)

    # Activity by Hour chart
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    ax2.bar(activity_by_hour.index, activity_by_hour.values, color='salmon')
    ax2.set_title("User Activity by Hour of Day")
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Number of Visits')

    canvas2 = FigureCanvasTkAgg(fig2, master=frame)
    canvas2.get_tk_widget().grid(row=4, column=0, padx=10, pady=10)
    description2 = "This chart shows the peak hours when the user is most active on websites."

    label_desc2 = Label(frame, text=description2, font=("Arial", 10), wraplength=700)
    label_desc2.grid(row=5, column=0, padx=10, pady=10)

    # Update scroll region after the frame is created
    frame.update_idletasks()
    canvas.config(scrollregion=canvas.bbox("all"))

    window.mainloop()

# Running the UI to display the results
show_ui()
