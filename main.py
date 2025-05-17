import re
from datetime import datetime
import pandas as pd
from collections import Counter
import emoji # For emoji analysis
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# --- Configuration ---
CHAT_FILE_PATH = 'roni.txt'  # File path

# The script will try to auto-detect the top 2 senders.
# You can manually override them if needed, e.g.:
ACTUAL_USER1_NAME_IN_CHAT = None # Set to actual name from chat if auto-detection is not preferred
ACTUAL_USER2_NAME_IN_CHAT = None # Set to actual name from chat

# How they appear in the dashboard
DISPLAY_NAME_USER1 = "Roni"
DISPLAY_NAME_USER2 = "Tiff"
COLOR_USER1 = 'cornflowerblue' # Blue for "Him" in example
COLOR_USER2 = 'lightpink'      # Pink for "Her" in example
COLOR_TOTAL = 'mediumseagreen' # Green for "Us" in example

# For "Words and Emojis" - specific word tracking (like "love" in the example)
# Add words in lowercase. The analysis will be case-insensitive.
TRACKED_KEYWORDS = {
    "yes": ["yes", "yeah", "yep", "yea", "ok", "okay", "okk", "okkk", "sure", "alright"],
    "no": ["no", "nope", "nah"],
    "good": ["good", "great", "nice", "awesome", "amazing", "wonderful"],
    "hi": ["hi", "hello", "hey", "heya", "greetings"]
}

# Basic English stop words to exclude from top words list
STOP_WORDS = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
    'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
    'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
    'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 've', 'll', 'm', 're'
    # More tags
    , 'image', 'omitted', 'audio', 'video', 'sticker', 'gif', 'media', 'okay', 'yeah', 'nice', 'like', 'good' 
    , 'yes', 'im', 'ok', 'okk', 'okkk'
])


def parse_whatsapp_chat(filepath):
    """
    Parses a WhatsApp chat log file.
    Handles multi-line messages and various date formats.
    """
    messages_data = []
    # Regex to capture date, time, sender, and message.
    # Handles M/D/YY or M/D/YYYY, and H:MM or I:MM AM/PM
    message_pattern = re.compile(
        r"\[(\d{1,2}/\d{1,2}/\d{2,4}), (\d{2}:\d{2}:\d{2})\] ([^:]+): (.*)"
    )
    current_message_parts = None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                match = message_pattern.match(line)
                if match:
                    if current_message_parts:
                        messages_data.append(current_message_parts)

                    date_str, time_str, sender, message_content = match.groups()

                    # Different datetime format
                    # dt_combined_str = f"{date_str} {time_str.upper()}" # Ensure AM/PM is uppercase
                    # timestamp = None
                    # for fmt in ('%d/%m/%y %H:%M', '%d/%m/%y %I:%M %p',         # 2-digit year
                    #             '%d/%m/%Y %H:%M', '%d/%m/%Y %I:%M %p'):       # 4-digit year
                    #     try:
                    #         timestamp = datetime.strptime(dt_combined_str, fmt)
                    #         break
                    #     except ValueError:
                    #         continue

                    dt_combined_str = f"{date_str} {time_str}"
                    timestamp = None

                    for fmt in ('%m/%d/%y %H:%M:%S', '%m/%d/%Y %H:%M:%S'):
                        try:
                            timestamp = datetime.strptime(dt_combined_str, fmt)
                            break
                        except ValueError:
                            continue
                    # print(match.groups())
                    if timestamp:
                        current_message_parts = {
                            "timestamp": timestamp,
                            "sender": sender.strip(),
                            "message": message_content.strip()
                        }
                    else:
                        # print(f"Warning: Could not parse date-time: {dt_combined_str} in line: {line[:50]}...")
                        current_message_parts = None # Reset if timestamp fails
                        continue # Skip this line start

                elif current_message_parts and line: # Append to multi-line message
                    current_message_parts["message"] += "\n" + line
                # else: it's an empty line or a system message we're not capturing (e.g., "You created this group")

            if current_message_parts: # Add the last message
                messages_data.append(current_message_parts)
    
    except FileNotFoundError:
        print(f"Error: Chat file not found at {filepath}")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred during parsing: {e}")
        return pd.DataFrame()

    if not messages_data:
        print("No messages parsed. Check file format or content.")
        return pd.DataFrame()

    df = pd.DataFrame(messages_data)
    if not df.empty:
        df = df.sort_values(by="timestamp").reset_index(drop=True)
        # Filter out common system messages if sender names are known or pattern for system messages.
        # For now, we assume senders are actual users.
    return df


def analyze_chat_data(df, user1_actual_name, user2_actual_name):
    """
    Performs various analyses on the parsed chat DataFrame.
    """
    if df.empty:
        return {}

    analysis = {}

    # Identify participants
    detected_senders = df['sender'].value_counts()
    if not user1_actual_name and len(detected_senders) > 0:
        user1_actual_name = detected_senders.index[0]
    if not user2_actual_name and len(detected_senders) > 1:
        user2_actual_name = detected_senders.index[1]
    elif not user2_actual_name and user1_actual_name and len(detected_senders) > 0 and detected_senders.index[0] != user1_actual_name : # if user1 was set, pick another
        user2_actual_name = detected_senders.index[0]
    elif not user2_actual_name and len(detected_senders) == 1 and detected_senders.index[0] == user1_actual_name:
        print("Warning: Only one primary sender detected or configured. Some comparative stats might be skewed.")
        user2_actual_name = "Other" # Placeholder if only one main user
    
    analysis['user1_actual_name'] = user1_actual_name
    analysis['user2_actual_name'] = user2_actual_name
    analysis['participants'] = [user1_actual_name, user2_actual_name] if user2_actual_name != "Other" else [user1_actual_name]


    # Basic Stats
    analysis['total_messages'] = len(df)
    analysis['chat_duration_days'] = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).days if analysis['total_messages'] > 0 else 0
    
    df['word_count'] = df['message'].apply(lambda x: len(x.split()))
    analysis['total_words'] = df['word_count'].sum()
    analysis['avg_words_per_message'] = analysis['total_words'] / analysis['total_messages'] if analysis['total_messages'] > 0 else 0
    
    all_words_flat = [word.lower() for msg in df['message'] for word in msg.split()]
    analysis['avg_word_length'] = sum(len(word) for word in all_words_flat) / len(all_words_flat) if all_words_flat else 0

    # Message distribution
    analysis['messages_per_participant'] = df['sender'].value_counts()

    # Temporal Analysis
    df['month_year_dt'] = df['timestamp'].dt.to_period('M').dt.to_timestamp() # For plotting
    analysis['monthly_counts_total'] = df.groupby('month_year_dt').size()
    
    # Ensure only the two main participants are explicitly columns, others are summed if necessary
    # For simplicity, focus on the two main participants only for this breakdown.
    # Filter df for main participants before grouping for per-participant monthly counts
    main_participants_df = df[df['sender'].isin(analysis['participants'])]
    analysis['monthly_counts_per_participant'] = main_participants_df.groupby(['month_year_dt', 'sender']).size().unstack(fill_value=0)
    
    if user1_actual_name not in analysis['monthly_counts_per_participant'].columns and user1_actual_name:
        analysis['monthly_counts_per_participant'][user1_actual_name] = 0
    if user2_actual_name not in analysis['monthly_counts_per_participant'].columns and user2_actual_name:
        analysis['monthly_counts_per_participant'][user2_actual_name] = 0


    analysis['cumulative_monthly_total'] = analysis['monthly_counts_total'].cumsum()
    analysis['cumulative_monthly_per_participant'] = analysis['monthly_counts_per_participant'].cumsum()
    
    df['weekday'] = df['timestamp'].dt.day_name()
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    analysis['weekday_counts'] = df['weekday'].value_counts().reindex(days_order, fill_value=0)
    
    df['hour'] = df['timestamp'].dt.hour
    analysis['hour_counts'] = df['hour'].value_counts().sort_index()
    analysis['hour_counts'] = analysis['hour_counts'].reindex(range(24), fill_value=0) # Ensure all hours 0-23 are present

    # Content Analysis
    all_text_for_words = ' '.join(df['message']).lower()
    words = re.findall(r'\b[a-z]+\b', all_text_for_words) # Only alphabetic words
    filtered_words = [word for word in words if word not in STOP_WORDS and len(word) > 1]
    analysis['top_words'] = Counter(filtered_words).most_common(30)

    # all_emojis_found = []
    # for message_text in df["message"]:
    #     emojis_in_message = emoji.emoji_list(message_text)  # Use emoji.emoji_list to extract emojis
    #     for emoji_item in emojis_in_message:
    #         all_emojis_found.append(emoji_item['emoji'])  # Extract the emoji from the dictionary
    # analysis['top_emojis'] = Counter(all_emojis_found).most_common(15)

    # Tracked Keywords Analysis
    analysis['tracked_keywords_counts'] = {}
    for category, keywords_list in TRACKED_KEYWORDS.items():
        category_counts_user1 = 0
        category_counts_user2 = 0
        for _, row in df.iterrows():
            msg_lower = row['message'].lower()
            is_user1 = row['sender'] == user1_actual_name
            is_user2 = row['sender'] == user2_actual_name
            
            for keyword_variant in keywords_list:
                if emoji.is_emoji(keyword_variant): # If the "keyword" is an emoji
                    count_in_msg = msg_lower.count(keyword_variant) # Simple count for emojis
                else: # If it's a word
                    # Use regex to count whole word occurrences
                    count_in_msg = len(re.findall(r'\b' + re.escape(keyword_variant) + r'\b', msg_lower))

                if count_in_msg > 0:
                    if is_user1:
                        category_counts_user1 += count_in_msg
                    elif is_user2:
                        category_counts_user2 += count_in_msg
        analysis['tracked_keywords_counts'][category] = {
            user1_actual_name: category_counts_user1,
            user2_actual_name: category_counts_user2
        }


    # Question Analysis
    question_starters = ['what', 'when', 'where', 'who', 'how', 'why', 'is', 'are', 'do', 'does', 'did', 'can', 'could', 'will', 'would', 'should']
    q_data = []
    for _, row in df.iterrows():
        msg_lower = row['message'].lower()
        # Check if ends with question mark or starts with a question word and is short (heuristic)
        is_q = msg_lower.endswith('?')
        if not is_q:
            first_word = msg_lower.split(' ')[0].strip('?.,!')
            if first_word in question_starters and len(msg_lower.split()) < 10 : # Heuristic for implicit questions
                 is_q = True

        if is_q:
            tokens = re.findall(r'\b\w+\b', msg_lower)
            for token in tokens:
                if token in question_starters: # Count occurrences of known question words within questions
                    q_data.append({'sender': row['sender'], 'q_word': token})
    
    if q_data:
        q_df = pd.DataFrame(q_data)
        analysis['question_word_freq'] = q_df.groupby(['q_word', 'sender']).size().unstack(fill_value=0)
        # Summing up to get overall frequency per q_word for general plot
        analysis['question_word_overall_freq'] = q_df['q_word'].value_counts()
    else:
        analysis['question_word_freq'] = pd.DataFrame()
        analysis['question_word_overall_freq'] = pd.Series(dtype='int')
        
    return analysis

# --- Plotting Functions ---
def plot_overall_stats(ax, data, user1_name, user2_name):
    ax.set_axis_off()
    duration_years = data['chat_duration_days'] / 365.25
    title = f"{duration_years:.1f}+ YEARS OF CHAT HISTORY" if duration_years >=1 else f"{data['chat_duration_days']} DAYS OF CHAT HISTORY"
    ax.text(0.5, 0.90, title, ha='center', va='center', fontsize=16, weight='bold')

    stats_text = (
        f"Total Messages: {data['total_messages']:,}\n"
        f"Total Words: {data['total_words']:,}\n"
        f"Avg. Words/Message: {data['avg_words_per_message']:.1f}\n"
        f"Avg. Word Length: {data['avg_word_length']:.2f}"
    )
    ax.text(0.5, 0.45, stats_text, ha='center', va='center', fontsize=11, linespacing=1.8)

def plot_message_distribution_pie(ax, data, user1_actual, user2_actual, user1_display, user2_display):
    ax.set_axis_off()
    counts = data['messages_per_participant']
    user1_count = counts.get(user1_actual, 0)
    user2_count = counts.get(user2_actual, 0)
    
    sizes = [user1_count, user2_count]
    labels = [f"{user1_display}\n({user1_count:,})", f"{user2_display}\n({user2_count:,})"]
    colors = [COLOR_USER1, COLOR_USER2]
    
    if sum(sizes) == 0: # No messages from these users
        ax.text(0.5, 0.5, "No messages from\nspecified users.", ha='center', va='center', fontsize=10, color='grey')
        return

    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90,
           wedgeprops={'edgecolor': 'white'}, textprops={'fontsize': 9})
    ax.set_title("Message Distribution", fontsize=12, pad=10)

def plot_monthly_messages(ax, data, user1_actual, user2_actual, user1_display, user2_display):
    if not data['monthly_counts_total'].empty:
        data['monthly_counts_total'].plot(ax=ax, label='Total', color=COLOR_TOTAL, marker='o', linestyle='-')
    
    monthly_per_participant = data['monthly_counts_per_participant']
    if user1_actual in monthly_per_participant.columns and not monthly_per_participant[user1_actual].empty:
        monthly_per_participant[user1_actual].plot(ax=ax, label=user1_display, color=COLOR_USER1, marker='.', linestyle='--')
    if user2_actual in monthly_per_participant.columns and not monthly_per_participant[user2_actual].empty:
        monthly_per_participant[user2_actual].plot(ax=ax, label=user2_display, color=COLOR_USER2, marker='.', linestyle='--')

    ax.set_title('Monthly Messages Over Time', fontsize=12)
    ax.set_xlabel('Month', fontsize=9)
    ax.set_ylabel('Number of Messages', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=':', alpha=0.7)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)


def plot_cumulative_messages(ax, data, user1_actual, user2_actual, user1_display, user2_display):
    if not data['cumulative_monthly_total'].empty:
        data['cumulative_monthly_total'].plot(ax=ax, label='Total Cumulative', color=COLOR_TOTAL, linewidth=2.5)
        
    cumulative_per_participant = data['cumulative_monthly_per_participant']
    if user1_actual in cumulative_per_participant.columns and not cumulative_per_participant[user1_actual].empty:
        cumulative_per_participant[user1_actual].plot(ax=ax, label=f'{user1_display} Cumulative', color=COLOR_USER1, linestyle=':')
    if user2_actual in cumulative_per_participant.columns and not cumulative_per_participant[user2_actual].empty:
        cumulative_per_participant[user2_actual].plot(ax=ax, label=f'{user2_display} Cumulative', color=COLOR_USER2, linestyle=':')

    ax.set_title('Total Messages Over Time (Cumulative)', fontsize=12)
    ax.set_xlabel('Month', fontsize=9)
    ax.set_ylabel('Cumulative Messages', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=':', alpha=0.7)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)


def plot_weekday_frequency(ax, data):
    if not data['weekday_counts'].empty:
        data['weekday_counts'].plot(kind='bar', ax=ax, color='skyblue', edgecolor='grey')
    ax.set_title('Message Frequency by Weekday', fontsize=12)
    ax.set_xlabel('')
    ax.set_ylabel('Number of Messages', fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.grid(axis='y', linestyle=':', alpha=0.7)

def plot_hour_frequency(ax, data):
    if not data['hour_counts'].empty:
        data['hour_counts'].plot(kind='barh', ax=ax, color='lightcoral', edgecolor='grey')
    ax.set_title('Message Frequency by Hour of Day', fontsize=12)
    ax.set_xlabel('Number of Messages', fontsize=9)
    ax.set_ylabel('Hour (24h)', fontsize=9)
    ax.set_yticks(range(0,24,2)) # Show every other hour tick
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(axis='x', linestyle=':', alpha=0.7)

# --- Plotting Functions ---
# ... (other plotting functions like plot_overall_stats, etc.) ...

# Ensure NUM_TOP_ITEMS is defined in your configuration section, e.g.:
# NUM_TOP_ITEMS = 30

# --- Plotting Functions ---
# ... (other plotting functions like plot_overall_stats, etc.) ...

def plot_top_items(ax, top_items_data, title, item_type="Words"):
    ax.set_axis_off()
    ax.text(0.5, 0.95, title, ha='center', va='top', fontsize=12, weight='bold')

    # top_items_data is already a list of (item, count) tuples,
    # its length is determined by NUM_TOP_ITEMS used in analyze_chat_data

    if not top_items_data:
        ax.text(0.5, 0.5, f"No {item_type.lower()} found.", ha='center', va='center', color='grey', fontsize=9)
        return

    font_size = 7 
    y_decrement_per_line = 0.031 
    available_height_ratio = 0.85 - 0.05 

    if y_decrement_per_line <= 0: # Safety check
        ax.text(0.5, 0.5, "Invalid spacing.", ha='center', va='center', color='red', fontsize=9)
        return

    # Calculate how many items can be displayed cleanly with the chosen font and spacing
    max_items_displayable = int(available_height_ratio / y_decrement_per_line)
    
    # We will plot the minimum of what's available and what can fit cleanly
    num_to_plot = min(len(top_items_data), max_items_displayable)

    if num_to_plot == 0 and len(top_items_data) > 0:
        # This means even one item couldn't fit, which implies y_decrement_per_line is too large
        # or available_height_ratio is too small. Or no items were passed.
        ax.text(0.5, 0.5, "Not enough space\nto display items.", ha='center', va='center', color='orange', fontsize=9)
        return
    elif num_to_plot == 0: # Handles the case from the initial check too
        ax.text(0.5, 0.5, f"No {item_type.lower()} found.", ha='center', va='center', color='grey', fontsize=9)
        return


    y_pos = 0.85  # Starting Y position for the first item (below the title)
    
    for i in range(num_to_plot):
        item, count = top_items_data[i]
        
        # The number of items to plot is already capped, so y_pos should not go out of bounds
        # if this logic is correct.

        ax.text(0.05, y_pos, f"{i+1}. {item}", ha='left', va='top', fontsize=font_size, clip_on=True)
        ax.text(0.95, y_pos, f"{count:,}", ha='right', va='top', fontsize=font_size, clip_on=True)
        y_pos -= y_decrement_per_line
        
    # If fewer items were plotted than requested due to space constraints, you could add a note
    if num_to_plot < len(top_items_data) and num_to_plot > 0 :
        # Add a small note at the bottom if list was truncated
        note_y_pos = y_pos # y_pos is now at the position for the *next* item
        if note_y_pos < 0.05 : note_y_pos = 0.05 # ensure it's within bounds
        # ax.text(0.5, note_y_pos, f"(showing top {num_to_plot} of {len(top_items_data)})", 
        #          ha='center', va='top', fontsize=font_size-1, color='grey', clip_on=True)
        pass # Optional: decide if you want to show a truncation note

def plot_tracked_keyword_pies(fig, base_gs_loc, data, user1_actual, user2_actual, user1_display, user2_display):
    """Plots small pie charts for tracked keywords."""
    num_keywords = len(TRACKED_KEYWORDS)
    if num_keywords == 0: return []
    
    # Create a sub-gridspec for these pies
    # e.g., if base_gs_loc is gs[5, :], this creates a grid within that span
    # This needs to be handled carefully in the main dashboard layout.
    # For now, let's assume we get an array of axes.
    
    axes_list = []
    keyword_data = data['tracked_keywords_counts']

    # Calculate rows/cols for keyword pies
    # Let's aim for a horizontal layout for keywords like in the example.
    # This function will be called with specific axes passed to it.
    # The caller needs to manage the GridSpec for these.

    for i, (category, counts) in enumerate(keyword_data.items()):
        # This function will now expect a single 'ax' for each keyword pie
        # and be called multiple times by the main dashboard code.
        # This means this function 'plot_tracked_keyword_pies' itself might be refactored
        # to plot ONE such pie on a given 'ax'.
        
        # Let's modify to plot one keyword pie on a given ax:
        # plot_single_keyword_pie(ax, category_name, counts_dict, user1_actual, ...)
        pass # This function will be simplified/called differently.


def plot_single_keyword_pie(ax, category_name, counts_dict, user1_actual, user2_actual, user1_display, user2_display):
    user1_count = counts_dict.get(user1_actual, 0)
    user2_count = counts_dict.get(user2_actual, 0)

    sizes = [user1_count, user2_count]
    labels = [f"{user1_display.split(' ')[0]}\n({user1_count})", f"{user2_display.split(' ')[0]}\n({user2_count})"] # Shorten display name
    colors_pie = [COLOR_USER1, COLOR_USER2]
    
    if sum(sizes) == 0:
        ax.text(0.5, 0.5, "N/A", ha='center', va='center', fontsize=8, color='grey')
    else:
        ax.pie(sizes, labels=labels, colors=colors_pie, autopct=lambda p: '{:.0f}'.format(p * sum(sizes) / 100), # Show actual counts
               startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth':0.5},
               textprops={'fontsize': 7}) # Smaller font for these pies
    ax.set_title(category_name.capitalize(), fontsize=9, pad=3)


def plot_question_analysis(ax, data, user1_actual, user2_actual, user1_display, user2_display):
    q_overall_freq = data.get('question_word_overall_freq')
    if q_overall_freq is None or q_overall_freq.empty:
        ax.text(0.5, 0.5, "No question data.", ha='center', va='center', color='grey')
        ax.set_axis_off()
        return

    # For simplicity, plotting overall frequency of question words
    top_q_words = q_overall_freq.nlargest(7) # Top 7 question words
    if not top_q_words.empty:
        top_q_words.plot(kind='barh', ax=ax, color='teal', edgecolor='grey')
    ax.set_title('Top Question Starter Words Used', fontsize=12)
    ax.set_xlabel('Frequency', fontsize=9)
    ax.set_ylabel('')
    ax.tick_params(labelsize=8)
    ax.invert_yaxis()
    ax.grid(axis='x', linestyle=':', alpha=0.7)
    # To show per user (more complex for this bar chart, could be stacked or grouped)
    # q_user_freq = data.get('question_word_freq')
    # if q_user_freq is not None and not q_user_freq.empty:
    #    q_user_freq.loc[top_q_words.index].plot(kind='barh', ax=ax, stacked=True, color=[COLOR_USER1, COLOR_USER2])
    #    ax.legend([user1_display, user2_display], fontsize=8)


# --- Main Dashboard Creation ---
def create_dashboard(data, user1_actual, user2_actual, user1_display, user2_display):
    if not data or not data.get('total_messages'):
        print("Not enough data to create dashboard.")
        plt.figure(figsize=(10,2))
        plt.text(0.5,0.5, "Not enough data to create dashboard.\nCheck chat file or parsing.", ha='center', va='center')
        plt.gca().set_axis_off()
        plt.show()
        return

    # Define layout based on the infographic
    # (This will be a tall figure)
    fig = plt.figure(figsize=(14, 28)) # Increased height
    # Grid: ~7 main rows, 2 columns for some
    gs = gridspec.GridSpec(8, 4, figure=fig, hspace=0.9, wspace=0.5,
                        height_ratios=[1.5, 2, 3, 3, 2.5, 5, 2, 2.5])

    # Row 0: Overall Stats (Spans 2 cols) + Message Pie (Spans 2 cols)
    ax_overall_stats = fig.add_subplot(gs[0, :2])
    plot_overall_stats(ax_overall_stats, data, user1_display, user2_display)

    ax_message_pie = fig.add_subplot(gs[0, 2:])
    plot_message_distribution_pie(ax_message_pie, data, user1_actual, user2_actual, user1_display, user2_display)

    # Row 1: Monthly Messages (Spans all 4 cols)
    ax_monthly = fig.add_subplot(gs[1, :])
    plot_monthly_messages(ax_monthly, data, user1_actual, user2_actual, user1_display, user2_display)

    # Row 2: Cumulative Messages (Spans all 4 cols)
    ax_cumulative = fig.add_subplot(gs[2, :])
    plot_cumulative_messages(ax_cumulative, data, user1_actual, user2_actual, user1_display, user2_display)

    # Row 3: Weekday Freq (Spans 2 cols) + Hour Freq (Spans 2 cols)
    ax_weekday = fig.add_subplot(gs[3, :2])
    plot_weekday_frequency(ax_weekday, data)

    ax_hour = fig.add_subplot(gs[3, 2:])
    plot_hour_frequency(ax_hour, data)
    
    # Row 4: Titles for Top Words/Emojis (Full Span)
    ax_content_title = fig.add_subplot(gs[4, :])
    ax_content_title.set_axis_off()
    ax_content_title.text(0.5, 0.7, "Lexicon & Expressions", fontsize=16, weight='bold', ha='center')
    ax_content_title.text(0.5, 0.3, "(Top Words, Emojis, and Keyword Usage)", fontsize=10, ha='center')


    # Row 5: Top Words (Spans 2 cols) + Top Emojis (Spans 2 cols)
    ax_top_words = fig.add_subplot(gs[5, :2])
    plot_top_items(ax_top_words, data.get('top_words', []), "Top Words (excluding stop words)", "Words")

    ax_top_emojis = fig.add_subplot(gs[5, 2:])
    plot_top_items(ax_top_emojis, data.get('top_emojis', []), "Top Emojis", "Emojis")


    # Row 6: Small Keyword Pies (Distribute across 4 columns)
    # Determine how many keyword categories we have to plot
    num_keyword_categories = len(TRACKED_KEYWORDS)
    keyword_axes = []
    if num_keyword_categories > 0:
        cols_for_keywords = 4 # Use all 4 columns for this row
        
        # Dynamically create subplots for keywords in this row
        # Example for up to 4 keyword categories:
        keyword_categories_to_plot = list(data['tracked_keywords_counts'].keys())[:cols_for_keywords]

        for i, category_name in enumerate(keyword_categories_to_plot):
            ax_kw_pie = fig.add_subplot(gs[6, i]) # Place each pie in a new column of this row
            counts_for_category = data['tracked_keywords_counts'][category_name]
            plot_single_keyword_pie(ax_kw_pie, category_name, counts_for_category,
                                    user1_actual, user2_actual, user1_display, user2_display)
    else: # Placeholder if no keywords defined or found
        ax_no_kw = fig.add_subplot(gs[6,:])
        ax_no_kw.set_axis_off()
        ax_no_kw.text(0.5,0.5, "No tracked keyword data.", ha='center', va='center', color='grey')


    # Row 7: Question Analysis (Spans all 4 cols)
    ax_questions = fig.add_subplot(gs[7, :])
    plot_question_analysis(ax_questions, data, user1_actual, user2_actual, user1_display, user2_display)

    fig.suptitle("Chat Analysis Dashboard", fontsize=20, weight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust rect to make space for suptitle and bottom
    
    plt.savefig("chat_dashboard.png", dpi=300, bbox_inches='tight')
    print("Dashboard saved as chat_dashboard.png")
    plt.show()


if __name__ == '__main__':
    print(f"Parsing chat data from: {CHAT_FILE_PATH}")
    df_chat = parse_whatsapp_chat(CHAT_FILE_PATH)

    if not df_chat.empty:
        print(f"Parsed {len(df_chat)} messages.")
        
        # Auto-detect or use configured participant names
        detected_senders_vc = df_chat['sender'].value_counts()
        auto_user1 = detected_senders_vc.index[0] if len(detected_senders_vc) > 0 else "User1_Auto"
        auto_user2 = detected_senders_vc.index[1] if len(detected_senders_vc) > 1 else "User2_Auto"

        # Use configured names if provided, otherwise use auto-detected
        final_user1_actual = ACTUAL_USER1_NAME_IN_CHAT if ACTUAL_USER1_NAME_IN_CHAT else auto_user1
        final_user2_actual = ACTUAL_USER2_NAME_IN_CHAT if ACTUAL_USER2_NAME_IN_CHAT else auto_user2
        
        # Update display names if they are the default "User 1/2" to reflect actual names
        display_user1 = DISPLAY_NAME_USER1 if DISPLAY_NAME_USER1 != "User 1 (e.g., Him)" else final_user1_actual.split(' ')[0] # Use first name
        display_user2 = DISPLAY_NAME_USER2 if DISPLAY_NAME_USER2 != "User 2 (e.g., Her)" else final_user2_actual.split(' ')[0]


        print(f"Analyzing data for: {display_user1} (as {final_user1_actual}) and {display_user2} (as {final_user2_actual})...")
        analysis_results = analyze_chat_data(df_chat, final_user1_actual, final_user2_actual)
        
        print("Creating dashboard...")
        create_dashboard(analysis_results, final_user1_actual, final_user2_actual, display_user1, display_user2)
    else:
        print("No data to analyze. Exiting.")
