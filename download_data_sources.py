import requests
import os
import pandas as pd

import io
from datetime import date


def download_json(url: str, filename: str):
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if "data" in data:
            df = pd.DataFrame(data['data'])
        else:
            df = pd.DataFrame(data)
        

        df.to_csv(filename, index=False, encoding='utf-8')
        
        print(f"\n Sucess: Data downloaded and saved as'{os.path.abspath(filename)}'.")
        
    except requests.exceptions.HTTPError as errh:
        print(f"\n Error (HTTP): Status code: {response.status_code}")
    except requests.exceptions.ConnectionError as errc:
        print(f"\n Error (Verbindung): {errc}")
    except requests.exceptions.Timeout as errt:
        print(f"\n Error (Timeout): {errt}")
    except requests.exceptions.RequestException as err:
        print(f"\n Error: Unknown Error: {err}")
        
def download_byte(url: str, filename: str):
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()
        
        byte=response.content
        string_data=byte.decode('utf-8')
        data_io = io.StringIO(string_data)
        df = pd.read_csv(data_io)
        
        
        df.to_csv(filename, index=False, encoding='utf-8')
        
        print(f"\n Sucess: Data downloaded and saved as'{os.path.abspath(filename)}'.")
        
    except requests.exceptions.HTTPError as errh:
        print(f"\n Error (HTTP): Status code: {response.status_code}")
    except requests.exceptions.ConnectionError as errc:
        print(f"\n Error (Verbindung): {errc}")
    except requests.exceptions.Timeout as errt:
        print(f"\n Error (Timeout): {errt}")
    except requests.exceptions.RequestException as err:
        print(f"\n Error: Unknown Error: {err}")

def download_csv(url: str, filename: str):
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(filename, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk: # Filtert Keep-alive-Chunks heraus
                            f.write(chunk)
        
        print(f"\n Sucess: Data downloaded and saved as'{os.path.abspath(filename)}'.")
        
    except requests.exceptions.HTTPError as errh:
        print(f"\n Error (HTTP): Status code: {response.status_code}")
    except requests.exceptions.ConnectionError as errc:
        print(f"\n Error (Connection): {errc}")
    except requests.exceptions.Timeout as errt:
        print(f"\n Error (Timeout): {errt}")
    except requests.exceptions.RequestException as err:
        print(f"\n Error: Unknown Error: {err}")



if __name__ == "__main__":
    today_date = date.today().strftime('%Y-%m-%d')
    
    DOWNLOAD_URL = "https://energy-api.instrat.pl/api/prices/co2?all=1" 
    FILE_NAME = "prices_eu_ets_all.csv"

    download_json(DOWNLOAD_URL, FILE_NAME)
    
    GAS_URL=(
    f"https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23ebf3fb&chart_type=line&drp=0&fo=open%20sans&"
    f"graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=off&txtcolor=%23444444&ts=12&tts=12&width=1019&"
    f"nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=PNGASEUUSDM&scale=left&cosd=1990-01-01&"
    f"coed={today_date}&line_color=%230073e6&link_values=false&line_style=solid&mark_type=none&mw=3&lw=3&ost=-99999&oet=99999&"
    f"mma=0&fml=a&fq=Monthly&fam=avg&fgst=lin&fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date={today_date}&"
    f"revision_date={today_date}&nd=1990-01-01"
    )#"https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23ebf3fb&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=off&txtcolor=%23444444&ts=12&tts=12&width=1019&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=PNGASEUUSDM&scale=left&cosd=1990-01-01&coed=2025-06-01&line_color=%230073e6&link_values=false&line_style=solid&mark_type=none&mw=3&lw=3&ost=-99999&oet=99999&mma=0&fml=a&fq=Monthly&fam=avg&fgst=lin&fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date=2025-11-27&revision_date=2025-11-27&nd=1990-01-01"
    FILE_NAME="PNGASEUUSDM.csv"
    download_byte(GAS_URL,FILE_NAME)
    
    COAL_URL=(
        f"https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23ebf3fb&chart_type=line&drp=0&fo=open%20sans&"
        f"graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=off&txtcolor=%23444444&ts=12&tts=12&width=1019&"
        f"nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=PCOALAUUSDM&scale=left&cosd=1990-01-01&"
        f"coed={today_date}&line_color=%230073e6&link_values=false&line_style=solid&mark_type=none&mw=3&lw=3&ost=-99999&oet=99999&"
        f"mma=0&fml=a&fq=Monthly&fam=avg&fgst=lin&fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date={today_date}&"
        f"revision_date={today_date}&n"
    )#"https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23ebf3fb&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=off&txtcolor=%23444444&ts=12&tts=12&width=1019&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=PCOALAUUSDM&scale=left&cosd=1990-01-01&coed=2025-06-01&line_color=%230073e6&link_values=false&line_style=solid&mark_type=none&mw=3&lw=3&ost=-99999&oet=99999&mma=0&fml=a&fq=Monthly&fam=avg&fgst=lin&fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date=2025-11-27&revision_date=2025-11-27&n"
    FILE_NAME="PCOALAUUSDM.csv"
    download_byte(COAL_URL,FILE_NAME)
    
    euro_URL=(
    f"https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23ebf3fb&chart_type=line&drp=0&fo=open%20sans&"
    f"graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1019&"
    f"nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=DEXUSEU&scale=left&cosd=2020-11-21&"
    f"coed={today_date}&line_color=%230073e6&link_values=false&line_style=solid&mark_type=none&mw=3&lw=3&ost=-99999&oet=99999&"
    f"mma=0&fml=a&fq=Daily&fam=avg&fgst=lin&fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date={today_date}&"
    f"revision_date={today_date}&nd=1999-01-04"
    )
    FILE_NAME="DEXUSEU.csv"
    download_byte(euro_URL,FILE_NAME)
    
    
    elec_price_URL="https://storage.googleapis.com/emb-prod-bkt-publicdata/public-downloads/price/outputs/european_wholesale_electricity_price_data_monthly.csv"
    FILE_NAME="european_wholesale_electricity_price_data_monthly.csv"
    download_csv(elec_price_URL,FILE_NAME)
    
    api_key="ca47530b-2ef6-2afb-c88f-5d542f3f0c64"
    carbon_itensity_URL=f"https://api.ember-energy.org/v1/carbon-intensity/monthly?entity=EU&is_aggregate_entity=true&start_date=2000-01&include_all_dates_value_range=false&api_key={api_key}"
    FILE_NAME="ember_monthly_carbon-intensity - All electricity sources - EU.csv"
    download_json(carbon_itensity_URL,FILE_NAME)
    
    generation_URL=f"https://api.ember-energy.org/v1/electricity-generation/monthly?entity=EU&is_aggregate_entity=true&start_date=2000-01&is_aggregate_series=true&include_all_dates_value_range=false&api_key={api_key}"
    FILE_NAME="ember_monthly_electricity-generation - All electricity sources - EU - breakdown.csv"
    download_json(generation_URL,FILE_NAME)