import pandas as pd
import re
from io import StringIO
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import os

# Define headers for requests to mimic a browser
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

# Define output directory
output_dir = '.'  # Current directory
csv_filename = 'results.csv'

# Define table configurations
table_configs = {
    'standard': {
        'url_type': 'stats',
        'columns': ['Player', 'Nation', 'Pos', 'Squad', 'Age', 'MP', 'Starts', 'Min', 'Gls', 'Ast', 'CrdY', 'CrdR', 'xG', 'npxG', 'xAG', 'PrgC', 'PrgP', 'PrgR', 'Gls.1', 'Ast.1', 'xG.1', 'xAG.1'],
        'rename': {
            'Nation': 'Nation', 'Pos': 'Pos', 'Squad': 'Squad', 'Age': 'Age', 'MP': 'MP', 'Starts': 'Starts', 'Min': 'Min',
            'Gls': 'Gls', 'Ast': 'Ast', 'CrdY': 'CrdY', 'CrdR': 'CrdR', 'xG': 'xG', 'npxG': 'npxG', 'xAG': 'xAG',
            'PrgC': 'PrgC_standard', 'PrgP': 'PrgP_standard', 'PrgR': 'PrgR_standard',
            'Gls.1': 'Gls_per90', 'Ast.1': 'Ast_per90', 'xG.1': 'xG_per90', 'xAG.1': 'xAG_per90',
        }
    },
    'goalkeeping': {
        'url_type': 'keepers',
        'columns': ['Player', 'GA90', 'Save%', 'CS%', 'Save%.1'],
        'rename': {'GA90': 'GA90', 'Save%': 'Save%', 'CS%': 'CS%', 'Save%.1': 'PK Save%'}
    },
    'shooting': {
        'url_type': 'shooting',
        'columns': ['Player', 'SoT%', 'SoT/90', 'G/Sh', 'Dist'],
        'rename': {'SoT%': 'SoT%', 'SoT/90': 'SoT/90', 'G/Sh': 'G/Sh', 'Dist': 'Dist'}
    },
    'passing': {
        'url_type': 'passing',
        'columns': ['Player', 'Cmp', 'Cmp%', 'TotDist', 'Cmp%.1', 'Cmp%.2', 'Cmp%.3', 'KP', '1/3', 'PPA', 'CrsPA', 'PrgP'],
        'rename': {
            'Cmp': 'Total Cmp', 'Cmp%': 'Total Cmp%', 'TotDist': 'TotDist_passing',
            'Cmp%.1': 'Short Cmp%', 'Cmp%.2': 'Medium Cmp%', 'Cmp%.3': 'Long Cmp%',
            'KP': 'KP', '1/3': '1/3_passing', 'PPA': 'PPA', 'CrsPA': 'CrsPA', 'PrgP': 'PrgP_passing'
        }
    },
    'gca': {
        'url_type': 'gca',
        'columns': ['Player', 'SCA', 'SCA90', 'GCA', 'GCA90'],
        'rename': {'SCA': 'SCA', 'SCA90': 'SCA90', 'GCA': 'GCA', 'GCA90': 'GCA90'}
    },
    'defense': {
        'url_type': 'defense',
        'columns': ['Player', 'Tkl', 'TklW', 'Att', 'Lost', 'Blocks', 'Sh', 'Pass', 'Int'],
        'rename': {
            'Tkl': 'Tkl', 'TklW': 'TklW', 'Att': 'Att_defense', 'Lost': 'Lost_defense',
            'Blocks': 'Blocks', 'Sh': 'Sh_defense', 'Pass': 'Pass_defense', 'Int': 'Int'
        }
    },
    'possession': {
        'url_type': 'possession',
        'columns': ['Player', 'Touches', 'Def Pen', 'Def 3rd', 'Mid 3rd', 'Att 3rd', 'Att Pen', 'Att', 'Succ%', 'Tkld%', 'Carries', 'TotDist', 'PrgDist', 'PrgC', '1/3', 'CPA', 'Mis', 'Dis', 'Rec', 'PrgR'],
        'rename': {
            'Touches': 'Touches', 'Def Pen': 'Def Pen', 'Def 3rd': 'Def 3rd', 'Mid 3rd': 'Mid 3rd', 'Att 3rd': 'Att 3rd', 'Att Pen': 'Att Pen',
            'Att': 'Att (Take-Ons)', 'Succ%': 'Succ%', 'Tkld%': 'Tkld%', 'Carries': 'Carries', 'TotDist': 'TotDist (Carries)', 'PrgDist': 'PrgDist (Carries)',
            'PrgC': 'PrgC_possession', '1/3': '1/3 (Carries)', 'CPA': 'CPA', 'Mis': 'Mis', 'Dis': 'Dis', 'Rec': 'Rec', 'PrgR': 'PrgR (Receiving)'
        }
    },
    'misc': {
        'url_type': 'misc',
        'columns': ['Player', 'Fls', 'Fld', 'Off', 'Crs', 'Recov', 'Won', 'Lost', 'Won%'],
        'rename': {'Fls': 'Fls', 'Fld': 'Fld', 'Off': 'Off', 'Crs': 'Crs', 'Recov': 'Recov', 'Won': 'Won', 'Lost': 'Lost_misc', 'Won%': 'Won%'}
    },
}

def scrape_table(config):
    """
    Scrapes a specified table from FBref.com using Selenium, extracting player IDs and required columns.
    """
    url = f'https://fbref.com/en/comps/9/{config["url_type"]}/Premier-League-Stats'
    
    # Set up Selenium with Chrome
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument(f'user-agent={headers["User-Agent"]}')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-notifications')
    chrome_options.add_argument('--disable-gcm')
    chrome_options.add_argument('--disable-sync')
    chrome_options.add_argument('--disable-extensions')
    chrome_options.add_argument('--disable-background-networking')
    
    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        driver.get(url)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, 'stats_table'))
        )
        
        # Find the player stats table
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
        
        # Extract player IDs and names
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
        
        # Flatten multi-level columns
        df.columns = [col[1] for col in df.columns]
        print(f"Columns at {url}: {df.columns.tolist()}")
        
        # Disambiguate duplicate columns
        duplicate_columns = ['Gls', 'Ast', 'xG', 'xAG', 'Save%', 'Cmp', 'Cmp%', 'Tkl', 'PassLive', 'PassDead', 'TO', 'Sh', 'Fld', 'Def']
        new_columns = []
        col_counts = {}
        for col in df.columns:
            if col in duplicate_columns:
                col_counts[col] = col_counts.get(col, 0) + 1
                if col == 'Save%':
                    new_columns.append(f"{col}_overall" if col_counts[col] == 1 else f"{col}_pk")
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
                    new_columns.append(f"{col}_sca" if col_counts[col] == 1 else f"{col}_gca")
                else:
                    new_columns.append(f"{col}_total" if col_counts[col] == 1 else f"{col}_per90")
            else:
                new_columns.append(col)
        
        df.columns = new_columns
        print(f"Columns after disambiguating duplicates: {df.columns.tolist()}")
        
        # Create column mapping
        column_mapping = {}
        for expected_col in config['columns']:
            if expected_col == 'Save%':
                temp_col = 'Save%_overall'
            elif expected_col == 'Save%.1':
                temp_col = 'Save%_pk'
            elif expected_col == 'Cmp':
                temp_col = 'Cmp_total'
            elif expected_col == 'Cmp%':
                temp_col = 'Cmp%_total'
            elif expected_col == 'Cmp%.1':
                temp_col = 'Cmp%_short'
            elif expected_col == 'Cmp%.2':
                temp_col = 'Cmp%_medium'
            elif expected_col == 'Cmp%.3':
                temp_col = 'Cmp%_long'
            elif expected_col == 'Tkl':
                temp_col = 'Tkl_total'
            elif expected_col == 'Sh':
                temp_col = 'Sh_sca'
            elif expected_col == 'Fld':
                temp_col = 'Fld_sca'
            elif expected_col.endswith('.1'):
                base_col = expected_col[:-2]
                temp_col = f"{base_col}_per90"
            else:
                temp_col = f"{expected_col}_total" if expected_col in ['Gls', 'Ast', 'xG', 'xAG'] else expected_col
            
            if temp_col in df.columns:
                column_mapping[temp_col] = expected_col
            elif expected_col in df.columns:
                column_mapping[expected_col] = expected_col
        
        print(f"Column mapping at {url}: {column_mapping}")
        
        # Rename columns
        df = df.rename(columns=column_mapping)
        print(f"DataFrame columns after renaming: {df.columns.tolist()}")
        
        # Deduplicate columns before numeric conversion
        duplicate_cols = df.columns[df.columns.duplicated()].tolist()
        if duplicate_cols:
            print(f"Duplicate columns found: {duplicate_cols}")
        df = df.loc[:, ~df.columns.duplicated(keep='first')]
        print(f"Columns after deduplication: {df.columns.tolist()}")
        
        # Ensure Player column exists
        player_col = 'Player' if 'Player' in df.columns else next((col for col in df.columns if 'Player' in col), None)
        if not player_col:
            print(f"Player column missing at {url}")
            return None
        if player_col != 'Player':
            df = df.rename(columns={player_col: 'Player'})
        
        # Create DataFrame with player IDs and names
        id_df = pd.DataFrame({'Player_ID': player_ids, 'Player': player_names})
        id_df = id_df[id_df['Player_ID'].notna() & id_df['Player'].notna()]
        
        # Filter main DataFrame
        df = df[df['Player'].isin(id_df['Player'])]
        df = df.reset_index(drop=True)
        
        # Add Player_ID column
        if len(id_df) == len(df):
            df['Player_ID'] = id_df['Player_ID'].reset_index(drop=True)
        else:
            print(f"Length mismatch at {url}: {len(id_df)} valid player IDs, {len(df)} rows")
            return None
        
        # Select required columns
        selected_columns = ['Player_ID', 'Player'] + config['columns'][1:] if config['url_type'] == 'stats' else ['Player_ID'] + config['columns']
        selected_columns = list(dict.fromkeys(selected_columns))
        
        # Check for missing columns
        missing_cols = [col for col in selected_columns if col not in df.columns]
        if missing_cols:
            print(f"Missing columns at {url}: {missing_cols}")
            for col in missing_cols:
                df[col] = 'N/a'
        
        df = df[selected_columns]
        
        # Rename columns to final names
        df = df.rename(columns=config['rename'])
        
        # Preprocess Age column
        if 'Age' in df.columns:
            df['Age'] = df['Age'].apply(lambda x: str(x).split('-')[0] if isinstance(x, str) and '-' in x else x)
        
        # Define numeric columns
        non_numeric_cols = ['Player_ID', 'Player', 'Nation', 'Pos', 'Squad']
        numeric_columns = [col for col in df.columns if col not in non_numeric_cols and col in df.columns]
        
        # Convert numeric columns
        if not df.empty:
            for col in numeric_columns:
                if col in df.columns:
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except Exception as e:
                        print(f"Error converting column {col} to numeric: {e}")
                        df[col] = 'N/a'
                else:
                    print(f"Column {col} not found in DataFrame")
                    df[col] = 'N/a'
        
        print(f"Final DataFrame columns: {df.columns.tolist()}")
        
        return df
    
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None
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
else:
    print("Standard table not available, proceeding with available tables")
    df_standard = None

# Merge DataFrames
if df_standard is not None:
    merged_df = df_standard
else:
    merged_df = next(iter(dfs.values()), None)
    if merged_df is None:
        print("No valid DataFrames to merge. No CSV file will be generated.")
        exit(1)

for table_type in table_configs:
    if table_type != 'standard' and table_type in dfs:
        merge_columns = ['Player_ID'] + list(table_configs[table_type]['rename'].values())
        merge_columns = list(dict.fromkeys(merge_columns))
        available_cols = [col for col in merge_columns if col in dfs[table_type].columns]
        if len(available_cols) < len(merge_columns):
            print(f"Warning: Missing columns in {table_type} for merge: {[col for col in merge_columns if col not in available_cols]}")
        merged_df = pd.merge(merged_df, dfs[table_type][available_cols], on='Player_ID', how='left')

# Aggregate data
agg_dict = {
    'Player': 'first', 'Nation': 'first', 'Pos': 'first', 'Squad': lambda x: ', '.join(x.dropna().unique()),
    'Age': 'first', 'MP': 'sum', 'Starts': 'sum', 'Min': 'sum', 'Gls': 'sum', 'Ast': 'sum',
    'CrdY': 'sum', 'CrdR': 'sum', 'xG': 'sum', 'npxG': 'sum', 'xAG': 'sum',
    'PrgC_standard': 'sum', 'PrgP_standard': 'sum', 'PrgR_standard': 'sum',
    'Gls_per90': 'mean', 'Ast_per90': 'mean', 'xG_per90': 'mean', 'xAG_per90': 'mean',
    'GA90': 'mean', 'Save%': 'mean', 'CS%': 'mean', 'PK Save%': 'mean',
    'SoT%': 'mean', 'SoT/90': 'mean', 'G/Sh': 'mean', 'Dist': 'mean',
    'Total Cmp': 'sum', 'Total Cmp%': 'mean', 'TotDist_passing': 'sum',
    'Short Cmp%': 'mean', 'Medium Cmp%': 'mean', 'Long Cmp%': 'mean',
    'KP': 'sum', '1/3_passing': 'sum', 'PPA': 'sum', 'CrsPA': 'sum', 'PrgP_passing': 'sum',
    'SCA': 'sum', 'SCA90': 'mean', 'GCA': 'sum', 'GCA90': 'mean',
    'Tkl': 'sum', 'TklW': 'sum', 'Att_defense': 'sum', 'Lost_defense': 'sum',
    'Blocks': 'sum', 'Sh_defense': 'sum', 'Pass_defense': 'sum', 'Int': 'sum',
    'Touches': 'sum', 'Def Pen': 'sum', 'Def 3rd': 'sum', 'Mid 3rd': 'sum', 'Att 3rd': 'sum', 'Att Pen': 'sum',
    'Att (Take-Ons)': 'sum', 'Succ%': 'mean', 'Tkld%': 'mean',
    'Carries': 'sum', 'TotDist (Carries)': 'sum', 'PrgDist (Carries)': 'sum',
    'PrgC_possession': 'sum', '1/3 (Carries)': 'sum', 'CPA': 'sum', 'Mis': 'sum', 'Dis': 'sum',
    'Rec': 'sum', 'PrgR (Receiving)': 'sum',
    'Fls': 'sum', 'Fld': 'sum', 'Off': 'sum', 'Crs': 'sum', 'Recov': 'sum',
    'Won': 'sum', 'Lost_misc': 'sum', 'Won%': 'mean',
}

merged_df = merged_df.groupby('Player_ID').agg(agg_dict).reset_index(drop=True)
merged_df = merged_df[merged_df['Min'] > 90]

# Add First_Name for sorting
merged_df['First_Name'] = merged_df['Player'].apply(lambda x: x.split()[0] if isinstance(x, str) and x.strip() else 'Unknown')
merged_df = merged_df.sort_values(by='First_Name')
merged_df = merged_df.drop(columns=['First_Name'])

# Fill missing values
merged_df = merged_df.fillna("N/a")

# Define final column order
final_columns = [
    'Player', 'Nation', 'Squad', 'Pos', 'Age', 'MP', 'Starts', 'Min', 'Gls', 'Ast', 'CrdY', 'CrdR',
    'xG', 'npxG', 'xAG', 'PrgC_standard', 'PrgP_standard', 'PrgR_standard', 'Gls_per90', 'Ast_per90', 'xG_per90', 'xAG_per90',
    'GA90', 'Save%', 'CS%', 'PK Save%', 'SoT%', 'SoT/90', 'G/Sh', 'Dist',
    'Total Cmp', 'Total Cmp%', 'TotDist_passing', 'Short Cmp%', 'Medium Cmp%', 'Long Cmp%', 'KP', '1/3_passing', 'PPA', 'CrsPA', 'PrgP_passing',
    'SCA', 'SCA90', 'GCA', 'GCA90', 'Tkl', 'TklW', 'Att_defense', 'Lost_defense', 'Blocks', 'Sh_defense', 'Pass_defense', 'Int',
    'Touches', 'Def Pen', 'Def 3rd', 'Mid 3rd', 'Att 3rd', 'Att Pen', 'Att (Take-Ons)', 'Succ%', 'Tkld%',
    'Carries', 'TotDist (Carries)', 'PrgDist (Carries)', 'PrgC_possession', '1/3 (Carries)', 'CPA', 'Mis', 'Dis',
    'Rec', 'PrgR (Receiving)', 'Fls', 'Fld', 'Off', 'Crs', 'Recov', 'Won', 'Lost_misc', 'Won%'
]

# Ensure all columns exist
for col in final_columns:
    if col not in merged_df.columns:
        merged_df[col] = "N/a"

final_df = merged_df[final_columns]

# Enhanced function to process Nation column
def extract_uppercase(nation):
    if pd.isna(nation) or not isinstance(nation, str) or nation.strip() == "":
        return "N/a"
    uppercase = ''.join(char for char in nation if char.isupper())
    return uppercase if uppercase else "N/a"

# Apply function to remove lowercase letters from Nation column
final_df['Nation'] = final_df['Nation'].apply(extract_uppercase)

# Validate data quality
for col in final_columns:
    na_count = final_df[col].eq("N/a").sum()
    if na_count > len(final_df) * 0.5:  # Warn if more than 50% of a column is "N/a"
        print(f"Warning: Column {col} has {na_count} 'N/a' values ({na_count/len(final_df)*100:.1f}% of rows)")

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Construct full CSV path
csv_path = os.path.join(output_dir, csv_filename)

# Save to CSV
final_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

print(f"Data successfully saved to {csv_path}")
if failed_tables:
    print(f"Warning: The following tables failed to scrape: {failed_tables}")
