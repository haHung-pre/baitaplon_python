import pandas as pd
import re
from io import StringIO
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import time
import os

# Define headers for requests to mimic a browser
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

# Define table configurations with URL types, columns to extract, and renaming rules
table_configs = {
    'standard': {
        'url_type': 'stats',
        'columns': ['Player', 'Nation', 'Pos', 'Squad', 'Age', 'MP', 'Starts', 'Min', 'Gls', 'Ast', 'CrdY', 'CrdR', 'xG', 'npxG', 'xAG', 'PrgC', 'PrgP', 'PrgR', 'Gls.1', 'Ast.1', 'xG.1', 'xAG.1'],
        'rename': {
            'MP': 'MP',
            'Starts': 'Starts',
            'Min': 'Min',
            'Gls': 'Gls',
            'Ast': 'Ast',
            'CrdY': 'CrdY',
            'CrdR': 'CrdR',
            'xG': 'xG',
            'npxG': 'npxG',
            'xAG': 'xAG',
            'PrgC': 'PrgC',
            'PrgP': 'PrgP',
            'PrgR': 'PrgR',
            'Gls.1': 'Gls_per90',
            'Ast.1': 'Ast_per90',
            'xG.1': 'xG_per90',
            'xAG.1': 'xAG_per90',
        }
    },
    'goalkeeping': {
        'url_type': 'keepers',
        'columns': ['Player', 'GA90', 'Save%', 'CS%', 'Save%.1'],
        'rename': {
            'GA90': 'GA90',
            'Save%': 'Save%',
            'CS%': 'CS%',
            'Save%.1': 'PK Save%',
        }
    },
    'shooting': {
        'url_type': 'shooting',
        'columns': ['Player', 'SoT%', 'SoT/90', 'G/Sh', 'Dist'],
        'rename': {
            'SoT%': 'SoT%',
            'SoT/90': 'SoT/90',
            'G/Sh': 'G/Sh',
            'Dist': 'Dist',
        }
    },
    'passing': {
        'url_type': 'passing',
        'columns': ['Player', 'Cmp', 'Cmp%', 'TotDist', 'Cmp%.1', 'Cmp%.2', 'Cmp%.3', 'KP', '1/3', 'PPA', 'CrsPA', 'PrgP'],
        'rename': {
            'Cmp': 'Total Cmp',
            'Cmp%': 'Total Cmp%',
            'TotDist': 'TotDist',
            'Cmp%.1': 'Short Cmp%',
            'Cmp%.2': 'Medium Cmp%',
            'Cmp%.3': 'Long Cmp%',
            'KP': 'KP',
            '1/3': '1/3',
            'PPA': 'PPA',
            'CrsPA': 'CrsPA',
            'PrgP': 'PrgP',
        }
    },
    'gca': {
        'url_type': 'gca',
        'columns': ['Player', 'SCA', 'SCA90', 'GCA', 'GCA90'],
        'rename': {
            'SCA': 'SCA',
            'SCA90': 'SCA90',
            'GCA': 'GCA',
            'GCA90': 'GCA90',
        }
    },
    'defense': {
        'url_type': 'defense',
        'columns': ['Player', 'Tkl', 'TklW', 'Att', 'Lost', 'Blocks', 'Sh', 'Pass', 'Int'],
        'rename': {
            'Tkl': 'Tkl',
            'TklW': 'TklW',
            'Att': 'Att',
            'Lost': 'Lost',
            'Blocks': 'Blocks',
            'Sh': 'Sh',
            'Pass': 'Pass',
            'Int': 'Int',
        }
    },
    'possession': {
        'url_type': 'possession',
        'columns': ['Player', 'Touches', 'Def Pen', 'Def 3rd', 'Mid 3rd', 'Att 3rd', 'Att Pen', 'Att', 'Succ%', 'Tkld%', 'Carries', 'TotDist', 'PrgDist', 'PrgC', '1/3', 'CPA', 'Mis', 'Dis', 'Rec', 'PrgR'],
        'rename': {
            'Touches': 'Touches',
            'Def Pen': 'Def Pen',
            'Def 3rd': 'Def 3rd',
            'Mid 3rd': 'Mid 3rd',
            'Att 3rd': 'Att 3rd',
            'Att Pen': 'Att Pen',
            'Att': 'Att (Take-Ons)',
            'Succ%': 'Succ%',
            'Tkld%': 'Tkld%',
            'Carries': 'Carries',
            'TotDist': 'TotDist (Carries)',
            'PrgDist': 'PrgDist (Carries)',
            'PrgC': 'PrgC (Carries)',
            '1/3': '1/3 (Carries)',
            'CPA': 'CPA',
            'Mis': 'Mis',
            'Dis': 'Dis',
            'Rec': 'Rec',
            'PrgR': 'PrgR (Receiving)',
        }
    },
    'misc': {
        'url_type': 'misc',
        'columns': ['Player', 'Fls', 'Fld', 'Off', 'Crs', 'Recov', 'Won', 'Lost', 'Won%'],
        'rename': {
            'Fls': 'Fls',
            'Fld': 'Fld',
            'Off': 'Off',
            'Crs': 'Crs',
            'Recov': 'Recov',
            'Won': 'Won',
            'Lost': 'Lost',
            'Won%': 'Won%',
        }
    },
}

# Function to scrape a table from FBref.com using Selenium
def scrape_table(config):
    """
    Scrapes a specified table from FBref.com using Selenium, extracting player IDs and required columns.
    
    Args:
        config (dict): Configuration with URL type, columns, and renaming rules.
    
    Returns:
        pd.DataFrame: Processed DataFrame with selected and renamed columns, or None if table not found.
    """
    url = f'https://fbref.com/en/comps/9/{config["url_type"]}/Premier-League-Stats'
    
    # Set up Selenium with Chrome
    chrome_options = Options()
    chrome_options.add_argument('--headless')  # Run in headless mode
    chrome_options.add_argument(f'user-agent={headers["User-Agent"]}')
    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        driver.get(url)
        time.sleep(3)  # Wait for JavaScript to load
        
        # Find the player stats table by ID or class
        table = None
        try:
            table = driver.find_element(By.ID, f'stats_{config["url_type"]}_9')
        except:
            tables = driver.find_elements(By.CLASS_NAME, 'stats_table')
            for tbl in tables:
                if tbl.find_elements(By.CSS_SELECTOR, 'td[data-stat="player"]'):
                    table = tbl
                    break
        
        if not table:
            print(f"No player stats table found at {url}")
            return None
        
        # Extract HTML and parse with BeautifulSoup
        soup = BeautifulSoup(table.get_attribute('outerHTML'), 'html.parser')
        
        # Extract player IDs and player names from player column links
        player_ids = []
        player_names = []
        player_rows = soup.find_all('td', {'data-stat': 'player'})
        for td in player_rows:
            a_tag = td.find('a')
            if a_tag and 'href' in a_tag.attrs:
                href = a_tag['href']
                match = re.search(r'/players/(\w+)/', href)
                if match:
                    player_ids.append(match.group(1))
                    player_names.append(td.get_text(strip=True))
                else:
                    player_ids.append(None)
                    player_names.append(None)
            else:
                player_ids.append(None)
                player_names.append(None)
        
        # Read table into DataFrame
        try:
            html_str = str(soup)
            df = pd.read_html(StringIO(html_str), header=[0,1])[0]
        except ValueError as e:
            print(f"Error reading table at {url}: {e}")
            return None
        
        # Flatten multi-level columns, using second-level header
        df.columns = [col[1] for col in df.columns]
        
        # Debug: Print original column names
        print(f"Columns at {url}: {df.columns.tolist()}")
        
        # Disambiguate duplicate columns
        duplicate_columns = ['Gls', 'Ast', 'xG', 'xAG', 'Save%', 'Cmp', 'Cmp%', 'Tkl', 'PassLive', 'PassDead', 'TO', 'Sh', 'Fld', 'Def']
        new_columns = []
        col_counts = {}
        for col in df.columns:
            if col in duplicate_columns:
                col_counts[col] = col_counts.get(col, 0) + 1
                if col == 'Save%':
                    if col_counts[col] == 1:
                        new_columns.append(f"{col}_overall")
                    else:
                        new_columns.append(f"{col}_pk")
                elif col == 'Cmp':
                    if col_counts[col] == 1:
                        new_columns.append(f"{col}_total")
                    elif col_counts[col] == 2:
                        new_columns.append(f"{col}_short")
                    elif col_counts[col] == 3:
                        new_columns.append(f"{col}_medium")
                    else:
                        new_columns.append(f"{col}_long")
                elif col == 'Cmp%':
                    if col_counts[col] == 1:
                        new_columns.append(f"{col}_total")
                    elif col_counts[col] == 2:
                        new_columns.append(f"{col}_short")
                    elif col_counts[col] == 3:
                        new_columns.append(f"{col}_medium")
                    else:
                        new_columns.append(f"{col}_long")
                elif col in ['PassLive', 'PassDead', 'TO', 'Sh', 'Fld', 'Def']:
                    if col_counts[col] == 1:
                        new_columns.append(f"{col}_sca")
                    else:
                        new_columns.append(f"{col}_gca")
                else:
                    if col_counts[col] == 1:
                        new_columns.append(f"{col}_total")
                    else:
                        new_columns.append(f"{col}_per90")
            else:
                new_columns.append(col)
        
        # Update DataFrame columns
        df.columns = new_columns
        print(f"Columns after disambiguating duplicates: {df.columns.tolist()}")
        
        # Create a column mapping for renaming
        column_mapping = {}
        for expected_col in config['columns']:
            print(f"Mapping expected column: {expected_col}")
            if expected_col == 'Save%':
                temp_col = 'Save%_overall'
                if temp_col in df.columns:
                    column_mapping[temp_col] = expected_col
                    print(f"Mapped {temp_col} to {expected_col}")
            elif expected_col == 'Save%.1':
                temp_col = 'Save%_pk'
                if temp_col in df.columns:
                    column_mapping[temp_col] = expected_col
                    print(f"Mapped {temp_col} to {expected_col}")
            elif expected_col == 'Cmp':
                temp_col = 'Cmp_total'
                if temp_col in df.columns:
                    column_mapping[temp_col] = expected_col
                    print(f"Mapped {temp_col} to {expected_col}")
            elif expected_col == 'Cmp%':
                temp_col = 'Cmp%_total'
                if temp_col in df.columns:
                    column_mapping[temp_col] = expected_col
                    print(f"Mapped {temp_col} to {expected_col}")
            elif expected_col == 'Cmp%.1':
                temp_col = 'Cmp%_short'
                if temp_col in df.columns:
                    column_mapping[temp_col] = expected_col
                    print(f"Mapped {temp_col} to {expected_col}")
            elif expected_col == 'Cmp%.2':
                temp_col = 'Cmp%_medium'
                if temp_col in df.columns:
                    column_mapping[temp_col] = expected_col
                    print(f"Mapped {temp_col} to {expected_col}")
            elif expected_col == 'Cmp%.3':
                temp_col = 'Cmp%_long'
                if temp_col in df.columns:
                    column_mapping[temp_col] = expected_col
                    print(f"Mapped {temp_col} to {expected_col}")
            elif expected_col == 'Tkl':
                temp_col = 'Tkl_total'
                if temp_col in df.columns:
                    column_mapping[temp_col] = expected_col
                    print(f"Mapped {temp_col} to {expected_col}")
            elif expected_col == 'Sh':
                temp_col = 'Sh_sca'
                if temp_col in df.columns:
                    column_mapping[temp_col] = expected_col
                    print(f"Mapped {temp_col} to {expected_col}")
            elif expected_col == 'Fld':
                temp_col = 'Fld_sca'
                if temp_col in df.columns:
                    column_mapping[temp_col] = expected_col
                    print(f"Mapped {temp_col} to {expected_col}")
            elif expected_col.endswith('.1'):
                base_col = expected_col[:-2]
                temp_col = f"{base_col}_per90"
                if temp_col in df.columns:
                    column_mapping[temp_col] = expected_col
                    print(f"Mapped {temp_col} to {expected_col}")
            else:
                temp_col = f"{expected_col}_total" if expected_col in ['Gls', 'Ast', 'xG', 'xAG'] else expected_col
                if temp_col in df.columns:
                    column_mapping[temp_col] = expected_col
                    print(f"Mapped {temp_col} to {expected_col}")
                elif expected_col in df.columns:
                    column_mapping[expected_col] = expected_col
                    print(f"Mapped non-duplicate {expected_col} to {expected_col}")
        
        # Debug: Print column mapping
        print(f"Column mapping at {url}: {column_mapping}")
        
        # Rename columns to match expected names
        df = df.rename(columns=column_mapping)
        
        # Debug: Print DataFrame columns after renaming
        print(f"DataFrame columns after renaming: {df.columns.tolist()}")
        
        # Find the Player column
        player_col = None
        for col in df.columns:
            if 'Player' in col:
                player_col = col
                break
        
        if not player_col:
            print(f"Player column missing at {url}")
            return None
        
        # Rename the player column to 'Player' if necessary
        if player_col != 'Player':
            df = df.rename(columns={player_col: 'Player'})
        
        # Create a DataFrame with player IDs and names
        id_df = pd.DataFrame({'Player_ID': player_ids, 'Player': player_names})
        
        # Filter out rows without valid player IDs or names
        id_df = id_df[id_df['Player_ID'].notna() & id_df['Player'].notna()]
        
        # Filter the main DataFrame to match valid player rows
        df = df[df['Player'].isin(id_df['Player'])]
        
        # Ensure player_ids length matches DataFrame rows
        if len(id_df) != len(df):
            print(f"Length mismatch at {url}: {len(id_df)} valid player IDs, {len(df)} rows after filtering")
            return None
        
        # Add player ID column
        df = df.reset_index(drop=True)
        df['Player_ID'] = id_df['Player_ID'].reset_index(drop=True)
        
        # Select required columns
        if config['url_type'] == 'stats':
            selected_columns = ['Player_ID', 'Player'] + config['columns'][1:]  # Include Player for standard stats
        else:
            selected_columns = ['Player_ID'] + config['columns']
        
        # Remove duplicates from selected_columns
        selected_columns = list(dict.fromkeys(selected_columns))
        
        # Check if all required columns exist
        missing_cols = [col for col in selected_columns if col not in df.columns]
        if missing_cols:
            print(f"Missing columns at {url}: {missing_cols}")
            # Instead of returning None, select available columns
            available_cols = [col for col in selected_columns if col in df.columns]
            if available_cols:
                df = df[available_cols]
            else:
                print(f"No required columns available at {url}")
                return None
        else:
            df = df[selected_columns]
        
        # Rename columns to simpler names
        df = df.rename(columns=config['rename'])
        
        # Debug: Print final DataFrame columns before deduplication
        print(f"Final DataFrame columns before deduplication: {df.columns.tolist()}")
        
        # Deduplicate final DataFrame columns (keep first occurrence)
        df = df.loc[:, ~df.columns.duplicated(keep='first')]
        
        # Debug: Print final DataFrame columns
        print(f"Final DataFrame columns: {df.columns.tolist()}")
        
        return df
    
    finally:
        driver.quit()

# Scrape all tables and store DataFrames
dfs = {}
failed_tables = []
for table_type, config in table_configs.items():
    df = scrape_table(config)
    if df is not None:
        dfs[table_type] = df
    else:
        failed_tables.append(table_type)
        print(f"Failed to scrape {table_type} table, continuing with other tables")

if not dfs:
    print("No tables were successfully scraped. No CSV file will be generated.")
    exit(1)

# Filter standard stats for players with more than 90 minutes
if 'standard' in dfs:
    df_standard = dfs['standard']
    df_standard['Min'] = pd.to_numeric(df_standard['Min'], errors='coerce')
    df_standard = df_standard[df_standard['Min'] > 90]
else:
    print("Standard table not available, proceeding with available tables")
    df_standard = None

# Merge all DataFrames on Player_ID
if df_standard is not None:
    merged_df = df_standard
else:
    # Use the first available table as the base
    merged_df = next(iter(dfs.values()), None)
    if merged_df is None:
        print("No valid DataFrames to merge. No CSV file will be generated.")
        exit(1)

for table_type in table_configs:
    if table_type != 'standard' and table_type in dfs:
        merge_columns = ['Player_ID'] + list(table_configs[table_type]['rename'].values())
        # Remove duplicates from merge_columns
        merge_columns = list(dict.fromkeys(merge_columns))
        available_cols = [col for col in merge_columns if col in dfs[table_type].columns]
        if len(available_cols) < len(merge_columns):
            print(f"Warning: Missing columns in {table_type} for merge: {[col for col in merge_columns if col not in available_cols]}")
        merged_df = pd.merge(merged_df, dfs[table_type][available_cols], on='Player_ID', how='left')

# Add First_Name column for sorting
merged_df['First_Name'] = merged_df['Player'].apply(lambda x: x.split()[0])

# Sort by First_Name
merged_df = merged_df.sort_values(by='First_Name')

# Drop temporary columns
merged_df = merged_df.drop(columns=['First_Name', 'Player_ID'])

# Fill missing values with "N/a"
merged_df = merged_df.fillna("N/a")

# Define final column order
final_columns = [
    'Player', 'Nation', 'Squad', 'Pos', 'Age', 'MP', 'Starts', 'Min', 'Gls', 'Ast', 'CrdY', 'CrdR', 
    'xG', 'npxG', 'xAG', 'PrgC', 'PrgP', 'PrgR', 'Gls_per90', 'Ast_per90', 'xG_per90', 'xAG_per90',
    'GA90', 'Save%', 'CS%', 'PK Save%', 'SoT%', 'SoT/90', 'G/Sh', 'Dist', 
    'Total Cmp', 'Total Cmp%', 'TotDist', 'Short Cmp%', 'Medium Cmp%', 'Long Cmp%', 'KP', '1/3', 'PPA', 'CrsPA', 'PrgP',
    'SCA', 'SCA90', 'GCA', 'GCA90', 'Tkl', 'TklW', 'Att', 'Lost', 'Blocks', 'Sh', 'Pass', 'Int',
    'Touches', 'Def Pen', 'Def 3rd', 'Mid 3rd', 'Att 3rd', 'Att Pen', 'Att (Take-Ons)', 'Succ%', 'Tkld%',
    'Carries', 'TotDist (Carries)', 'PrgDist (Carries)', 'PrgC (Carries)', '1/3 (Carries)', 'CPA', 'Mis', 'Dis',
    'Rec', 'PrgR (Receiving)', 'Fls', 'Fld', 'Off', 'Crs', 'Recov', 'Won', 'Lost', 'Won%'
]

# Ensure all final columns exist, fill missing with "N/a"
for col in final_columns:
    if col in merged_df.columns:
        merged_df[col] = merged_df[col].astype(str)
    else:
        merged_df[col] = "N/a"

# Select and order final columns
final_df = merged_df[final_columns]

# Save to CSV in the Downloads directory
csv_path = r'C:\Users\nguye\Downloads\results.csv'
final_df.to_csv(csv_path, index=False)

print(f"Data successfully saved to {csv_path}")
if failed_tables:
    print(f"Warning: The following tables failed to scrape: {failed_tables}")