import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
import time
import random
import subprocess
import csv
import logging
import os


def fetch_truspilot_reviews(site="www.oscaro.com",nb_pages=5):
    base_url = f"https://fr.trustpilot.com/review/{site}"
    all_data = []
    
    # Headers pour simuler un vrai navigateur
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7"
    }

    headers = {
         "authority":"api.segment.io"
        ,"method":"POST"
        ,"path":"/v1/m"
        ,"scheme":"https"
        ,"accept":"*/*"
        ,"accept-encoding":"gzip, deflate, br, zstd"
        ,"accept-language":"fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7"
        ,"content-length":"138"
        ,"content-type":"text/plain"
        ,"origin":"https://fr.trustpilot.com"
        ,"priority":"u=1, i"
        ,"referer":"https://fr.trustpilot.com/"
        ,"sec-ch-ua":'"Google Chrome";v="147", "Not.A/Brand";v="8", "Chromium";v="147"'
        ,"sec-ch-ua-mobile":"?0"
        ,"sec-ch-ua-platform":"Windows"
        ,"sec-fetch-dest":"empty"
        ,"sec-fetch-mode":"cors"
        ,"sec-fetch-site":"cross-site"
        ,"user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36"

    }
    
    for page in range(1, nb_pages + 1):
        print(f"Extraction de la page {page}...")
        
        # Gestion de l'URL pour la pagination
        url = f"{base_url}?page={page}"
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # On cherche la balise magique qui contient toutes les infos
            script_tag = soup.find('script', id='__NEXT_DATA__')
            
            if script_tag:
                data = json.loads(script_tag.string)
                # Parcours du dictionnaire JSON pour atteindre les avis
                reviews_list = data['props']['pageProps']['reviews']
                
                for rev in reviews_list:
                    all_data.append({
                        'Date': rev.get('createdAt'),
                        'Auteur': rev.get('consumer', {}).get('displayName'),
                        'Note': rev.get('rating'),
                        'Titre': rev.get('title'),
                        'Commentaire': rev.get('text'),
                        'Réponse_Oscaro': rev.get('reply', {}).get('message') if rev.get('reply') else "Pas de réponse"
                    })
            
            # Pause aléatoire pour ne pas être banni (comportement humain)
            time.sleep(random.uniform(1.5, 3.0))
            
        except Exception as e:
            print(f"Erreur sur la page {page}: {e}")
            break

    # Sauvegarde des données
    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv('avis_oscaro.csv', index=False, encoding='utf-16')
        print(f"\n✅ Succès ! {len(all_data)} avis enregistrés dans 'avis_oscaro.csv'.")
    else:
        print("❌ Aucune donnée n'a pu être extraite.")

# Lancement pour les 3 premières pages (tu peux changer le chiffre)
#fetch_truspilot_reviews(nb_pages=3)

def fetch_truspilot_reviews1(site="www.oscaro.com",nb_pages=5):
    return site,nb_pages

#Claude Analyse
def fetch_truspilot_reviews2(site="www.oscaro.com",nb_pages=3):

    # Ils exposent une API JSON utilisée par leur propre frontend
    url = "https://fr.trustpilot.com/api/categoriespages.jsonld/review/oscaro.com"

    # Ou mieux : l'endpoint reviews paginé
    url = "https://fr.trustpilot.com/review/oscaro.com"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "fr-FR,fr;q=0.9",
        "Referer": "https://fr.trustpilot.com/",
    }

    session = requests.Session()
    session.headers.update(headers)

    response = session.get(url, timeout=15)
    soup = BeautifulSoup(response.text, "html.parser")

    # Les données sont dans un script __NEXT_DATA__ (Next.js)
    import json

    script = soup.find("script", {"id": "__NEXT_DATA__"})
    data = json.loads(script.string)

    # Naviguer dans le JSON
    reviews = data["props"]["pageProps"]["reviews"]

    for r in reviews:
        print(r["text"])
        print(r["rating"])
        print(r["consumer"]["displayName"])
        print("---")

    return site




def fetch_truspilot_reviews3(site="www.oscaro.com",nb_pages=3):

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "fr-FR,fr;q=0.9",
        "Referer": "https://fr.trustpilot.com/",
    }
    
    session = requests.Session()
    session.headers.update(headers)

    response = session.get("https://fr.trustpilot.com/review/oscaro.com", timeout=15)

    print("Status:", response.status_code)
    print("Taille réponse:", len(response.text))
    print(response.text[:2000])  # Voir ce qu'on reçoit vraiment


global CURL_HEADERS, COOKIE_STRING, CSV_FILE, STATE_FILE, BASE_URL

def fetch_page(page,BASE_URL,CURL_HEADERS):
    """Fetch a page using curl (bypasses WAF issues with python requests)."""
    if page == 1:
        url = f"{BASE_URL}?businessUnit=oscaro.com"
        referer = "https://fr.trustpilot.com/review/oscaro.com"
    else:
        url = f"{BASE_URL}?page={page}&businessUnit=oscaro.com"
        referer = f"https://fr.trustpilot.com/review/oscaro.com?page={page - 1}"
    cmd = [
        "curl", "-s", "-f", url,
        "-b", COOKIE_STRING,
        "-H", f"referer: {referer}",
        *CURL_HEADERS,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"curl failed (exit {result.returncode}): {result.stderr.strip()}")
    return json.loads(result.stdout)


def flatten(obj, prefix=""):
    """Flatten a nested dict/list into dot-separated keys."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
            if isinstance(v, (dict, list)):
                out.update(flatten(v, key))
            else:
                out[key] = v
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            key = f"{prefix}[{i}]"
            if isinstance(v, (dict, list)):
                out.update(flatten(v, key))
            else:
                out[key] = v
    return out


def load_state(STATE_FILE):
    """Load set of completed pages and known CSV columns."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            data = json.load(f)
        return set(data.get("done_pages", [])), data.get("columns", [])
    return set(), []


def save_state(done_pages, columns, STATE_FILE):
    with open(STATE_FILE, "w") as f:
        json.dump({"done_pages": sorted(done_pages), "columns": columns}, f)

def _rewrite_csv_with_new_columns(columns, new_rows, CSV_FILE):
    """Re-read existing CSV, rewrite with expanded header, then append new rows."""
    existing_rows = []
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)

    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in existing_rows:
            writer.writerow(row)
        for row in new_rows:
            writer.writerow(row)

    log.info(
        "Rewrote CSV with %d columns (%d existing + %d new rows)",
        len(columns),
        len(existing_rows),
        len(new_rows),
    )


def fetch_truspilot_reviews4(site="www.oscaro.com",nb_pages=3, START_PAGE = 1):
    
    END_PAGE=START_PAGE+nb_pages-1

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    global log, CURL_HEADERS, COOKIE_STRING, CSV_FILE, STATE_FILE

    log = logging.getLogger(__name__)

    BASE_URL = "https://fr.trustpilot.com/_next/data/businessunitprofile-consumersite-2.6055.0/review/oscaro.com.json"

    CURL_HEADERS = [
        "-H", "accept: */*",
        "-H", "accept-language: fr,fr-FR;q=0.9,en-US;q=0.8,en;q=0.7",
        "-H", "baggage: sentry-environment=Production,sentry-release=businessunitprofile-consumersite%402.6055.0,sentry-public_key=7ac98d0742b24421b3d38448c4bf1184,sentry-trace_id=b93ec30c235842fea81da0fdff195187",
        "-H", "cache-control: no-cache",
        "-H", "newrelic: eyJ2IjpbMCwxXSwiZCI6eyJ0eSI6IkJyb3dzZXIiLCJhYyI6IjQzODU4ODciLCJhcCI6IjUzODY4MjQ5OSIsImlkIjoiZTJiZGMyMmQ0MDJmM2I0OCIsInRyIjoiZDJhMDhiOTM4OGMzNmQyNDcwY2RmNWE3YTdmZmFmMDIiLCJ0aSI6MTc3MTE3OTkyNDAxNH19",
        "-H", "pragma: no-cache",
        "-H", "priority: u=1, i",
        "-H", 'sec-ch-ua: "Not(A:Brand";v="8", "Chromium";v="144", "Google Chrome";v="144"',
        "-H", "sec-ch-ua-mobile: ?0",
        "-H", 'sec-ch-ua-platform: "macOS"',
        "-H", "sec-fetch-dest: empty",
        "-H", "sec-fetch-mode: cors",
        "-H", "sec-fetch-site: same-origin",
        "-H", "sentry-trace: b93ec30c235842fea81da0fdff195187-902fc00c9014fb8e",
        "-H", "traceparent: 00-d2a08b9388c36d2470cdf5a7a7ffaf02-e2bdc22d402f3b48-01",
        "-H", "tracestate: 4385887@nr=0-1-4385887-538682499-e2bdc22d402f3b48----1771179924014",
        "-H", "user-agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36",
        "-H", "x-nextjs-data: 1",
    ]
    COOKIE_STRING = "TP.uuid=e86ab5e8-0368-44ed-82a7-948368719bd5; OptanonAlertBoxClosed=2026-01-29T13:16:03.005Z; ajs_anonymous_id=7c20e80b-916f-460f-b2e4-d2bd5107e826; fc_referrer_url=https%3A%2F%2Fwww.google.com%2F; _ga=GA1.1.688628830.1769692563; _hjSessionUser_391767=eyJpZCI6IjJkNDIzN2ZmLTVmYzgtNTdhMC1iMTcxLTljYjg0YmVlMWVhNyIsImNyZWF0ZWQiOjE3Njk2OTI1NjMwOTYsImV4aXN0aW5nIjp0cnVlfQ==; amplitude_idundefinedtrustpilot.com=eyJvcHRPdXQiOmZhbHNlLCJzZXNzaW9uSWQiOm51bGwsImxhc3RFdmVudFRpbWUiOm51bGwsImV2ZW50SWQiOjAsImlkZW50aWZ5SWQiOjAsInNlcXVlbmNlTnVtYmVyIjowfQ==; lc_referrer_url=https%3A%2F%2Fwww.google.com%2F; _hjHasCachedUserAttributes=true; _hjSession_391767=eyJpZCI6ImExMzg3MWRhLWRhM2ItNGQzNy1iZWU0LWQ5YjQxZTM3YTNhOCIsImMiOjE3NzExNzk0NjM4MDEsInMiOjAsInIiOjAsInNiIjowLCJzciI6MCwic2UiOjAsImZzIjowLCJzcCI6MH0=; amplitude_id_0401371089d1a27b189b1976accb81fftrustpilot.com=eyJkZXZpY2VJZCI6IjdjMjBlODBiLTkxNmYtNDYwZi1iMmU0LWQyYmQ1MTA3ZTgyNiIsInVzZXJJZCI6bnVsbCwib3B0T3V0IjpmYWxzZSwic2Vzc2lvbklkIjoxNzcxMTc5ODEzOTc2LCJsYXN0RXZlbnRUaW1lIjoxNzcxMTc5ODE0MDI2LCJldmVudElkIjoyLCJpZGVudGlmeUlkIjoxLCJzZXF1ZW5jZU51bWJlciI6M30=; _hjSessionUser_386931=eyJpZCI6ImFmNjQ4ZWEzLTZiOWMtNTRlMy1iOGYxLWFjMDM3ZWYxZGM5ZCIsImNyZWF0ZWQiOjE3NzExNzk4MTQwOTcsImV4aXN0aW5nIjp0cnVlfQ==; _hjSession_386931=eyJpZCI6ImViOGE3OTI3LTlmNWEtNDgzMC1hYTExLTdlMmJjNjJjMTVjZSIsImMiOjE3NzExNzk4MTQwOTgsInMiOjEsInIiOjEsInNiIjowLCJzciI6MCwic2UiOjAsImZzIjoxLCJzcCI6MH0=; _uetsid=701145000a9b11f187d781796e3caa1d; _uetvid=70114d400a9b11f1aebfbde26c5b1e63; __adroll_fpc=46fd792eef4bda5045a5383e06d9c2dc-1771179814266; _fbp=fb.1.1771179815354.819193092275389343; _gcl_au=1.1.787948648.1769692563.1934256921.1771179815.1771179822; amplitude_id_cfe705a69359b8a4c0049d061ee5787btrustpilot.com=eyJkZXZpY2VJZCI6IjM3NTgxNDUzLTU5MGUtNDQyYy1iM2Y4LTU4YTcxMDExNTczOVIiLCJ1c2VySWQiOm51bGwsIm9wdE91dCI6ZmFsc2UsInNlc3Npb25JZCI6MTc3MTE3OTc2NTI4OCwibGFzdEV2ZW50VGltZSI6MTc3MTE3OTgzMDU1NSwiZXZlbnRJZCI6MjQsImlkZW50aWZ5SWQiOjksInNlcXVlbmNlTnVtYmVyIjozM30=; tp-external-referrer=%7B%22referrer%22%3A%22https%3A%2F%2Fwww.google.com%2F%22%2C%22entry%22%3A%22https%3A%2F%2Ffr.trustpilot.com%2Freview%2Foscaro.com%3Fpage%3D4%22%7D; OptanonConsent=isGpcEnabled=0&datestamp=Sun+Feb+15+2026+19%3A24%3A08+GMT%2B0100+(Central+European+Standard+Time)&version=202601.1.0&browserGpcFlag=0&isIABGlobal=false&hosts=&consentId=9d8926d8-8bfc-4ae7-8f8a-0e0ded074d4b&interactionCount=1&isAnonUser=1&landingPath=NotLandingPage&groups=C0001%3A1%2CC0002%3A1%2CC0003%3A1%2CC0004%3A1&intType=1&crTime=1769692563126&geolocation=FR%3BIDF&AwaitingReconsent=false; aws-waf-token=dd94a7df-ceef-4670-9fb4-64e8e5fd351e:DAoAdkOBJJ+AAAAA:qh3bQFJNOz/XQTdmZCGn+j2/GjOm0NE47DnKHLb6682kveM4NDE+a/PoyFT7H5Z255uDTq0LKRiqul0RJxKYVXbBdkheo5Cgjxkua+Z6IyfqByMjY0f1LxpqTVZnCaxrh2IYxvoF/+10+Q6mvip/XDPIjCgS20kzqCvbEyc7pT0PedqkTdqpy8vIWwCgjEVX2AbRDwUOM085RQkZYafqrDHEUkSyi6fcx8F/2/B8U+IsVoElR0jVLLBMOML4dhEkT+lOmjDK1Gc=; g_state={\"i_l\":0,\"i_ll\":1771179891955,\"i_e\":{\"enable_itp_optimization\":0}}; tp-consumer-id=69920f83d6c3522fd9542a03; jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjb25zdW1lcklkIjoiNjk5MjBmODNkNmMzNTIyZmQ5NTQyYTAzIiwiaGFzQWNjZXB0ZWRUZXJtcyI6dHJ1ZSwiaXNCbG9ja2VkRm9yUmVwb3J0aW5nIjpmYWxzZSwiYWNjZXNzVG9rZW4iOiJNUFBQYlZvNG51S0FBbG5mZXVCRmdENWpJR25WIiwiYXV0aGVudGljYXRpb25Tb3VyY2UiOiJnb29nbGUiLCJpYXQiOjE3NzExNzk5MTEsImV4cCI6MTc3ODk1NTkxMX0.J5XSljS1OZO2Bh8tPBioxTGtpVfSHV9hISHmBNUnHVg; ajs_user_id=79d14c79871998b0c9c7cd73eee6ad8ad166f750; amplitude_id_67f7b7e6c8cb1b558b0c5bda2f747b07trustpilot.com=eyJkZXZpY2VJZCI6IjdjMjBlODBiLTkxNmYtNDYwZi1iMmU0LWQyYmQ1MTA3ZTgyNiIsInVzZXJJZCI6Ijc5ZDE0Yzc5ODcxOTk4YjBjOWM3Y2Q3M2VlZTZhZDhhZDE2NmY3NTAiLCJvcHRPdXQiOmZhbHNlLCJzZXNzaW9uSWQiOjE3NzExNzk0NjM5MDIsImxhc3RFdmVudFRpbWUiOjE3NzExNzk5MTE5NTcsImV2ZW50SWQiOjQ4LCJpZGVudGlmeUlkIjo5LCJzZXF1ZW5jZU51bWJlciI6NTd9; _ga_11HBWMC274=GS2.1.s1771179464$o4$g1$t1771179913$j24$l0$h0"

    CSV_FILE = "../data/reviews.csv"
    STATE_FILE = "../data/.scrape_state.json"
    
    DELAY = 1.0

    done_pages, known_columns = load_state()
    print(STATE_FILE)

    if done_pages:
        log.info(
            "Resuming — %d pages already done, %d columns known",
            len(done_pages),
            len(known_columns),
        )

    if not known_columns:
        log.info("Fetching page 1 to discover CSV columns...")
        data = fetch_page(1,BASE_URL)
        reviews = data.get("pageProps", {}).get("reviews", [])
        if not reviews:
            log.error("No reviews on page 1 — cannot discover columns. Aborting.")
            log.error("Response keys: %s", list(data.get("pageProps", {}).keys()))
            return

        col_set = set()
        for r in reviews:
            col_set.update(flatten(r).keys())
        known_columns = ["_page"] + sorted(col_set)

        with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=known_columns, extrasaction="ignore")
            writer.writeheader()
            for r in reviews:
                row = flatten(r)
                row["_page"] = 1
                writer.writerow(row)

        done_pages.add(1)
        save_state(done_pages, known_columns)
        log.info("Page 1: %d reviews, %d columns discovered", len(reviews), len(known_columns))
        time.sleep(DELAY)

    # ---- main loop ----
    total_new = 0
    consecutive_errors = 0
    max_consecutive_errors = 10

    for page in range(START_PAGE, END_PAGE + 1):
        if page in done_pages:
            continue

        try:
            data = fetch_page(page,BASE_URL)
            reviews = data.get("pageProps", {}).get("reviews", [])

            if not reviews:
                log.info("Page %d: 0 reviews — reached the end", page)
                done_pages.add(page)
                save_state(done_pages, known_columns)
                break

            new_cols = set()
            rows = []
            for r in reviews:
                flat = flatten(r)
                flat["_page"] = page
                new_cols.update(flat.keys())
                rows.append(flat)

            unseen = new_cols - set(known_columns)
            if unseen:
                log.info("Page %d: found %d new columns: %s", page, len(unseen), unseen)
                known_columns.extend(sorted(unseen))
                _rewrite_csv_with_new_columns(known_columns, rows)
            else:
                with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(
                        f, fieldnames=known_columns, extrasaction="ignore"
                    )
                    for row in rows:
                        writer.writerow(row)

            total_new += len(reviews)
            done_pages.add(page)
            save_state(done_pages, known_columns)
            consecutive_errors = 0

            remaining = END_PAGE - page
            log.info(
                "Page %d: +%d reviews (total new: %d, done: %d/%d, remaining: %d)",
                page,
                len(reviews),
                total_new,
                len(done_pages),
                END_PAGE,
                remaining,
            )

        except Exception as e:
            log.error("Page %d: %s", page, e)
            consecutive_errors += 1

        if consecutive_errors >= max_consecutive_errors:
            log.error("Too many consecutive errors (%d). Stopping.", consecutive_errors)
            break

        time.sleep(DELAY)

    log.info(
        "Finished — %d pages done, %d new reviews this run. CSV: %s",
        len(done_pages),
        total_new,
        CSV_FILE,
    )

def Cookies2python(Cookie):
    res={}
    for u in Cookie.split("; "):
        #print(u, u.split("=",1))
        [key,val] = u.split("=",1)
        #print(key,val)
        res[key]=val
    return res


def testing():
    from IPython.core.display import display, HTML
       
    #from copy past dev tool - changing frequently

    Cookie="OptanonAlertBoxClosed=2026-01-06T08:59:54.335Z; jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjb25zdW1lcklkIjoiNjM3Yzk4Nzc0MjViZTMwMDEyNDc0ODllIiwiaGFzQWNjZXB0ZWRUZXJtcyI6dHJ1ZSwiaXNCbG9ja2VkRm9yUmVwb3J0aW5nIjpmYWxzZSwiYWNjZXNzVG9rZW4iOiJCVzF0cmVMYUltQm9CTHBjY2xDdm1COHRCNmdHIiwiYXV0aGVudGljYXRpb25Tb3VyY2UiOiJnb29nbGUiLCJpYXQiOjE3NzA0Mzk4NTksImV4cCI6MTc3ODIxNTg1OX0.mSSYeVlqklfPJidPFTDviljdAX0B7ziyu0n2PpCfEFs; _csrf=PiV8vIH9OBuWfdwWMfaxgfrC; _hjHasCachedUserAttributes=true; analytics_session_id=1776938089235; analytics_session_id.last_access=1776938437995; amplitude_idundefinedtrustpilot.com=eyJvcHRPdXQiOmZhbHNlLCJzZXNzaW9uSWQiOm51bGwsImxhc3RFdmVudFRpbWUiOm51bGwsImV2ZW50SWQiOjAsImlkZW50aWZ5SWQiOjAsInNlcXVlbmNlTnVtYmVyIjowfQ==; _twpid=tw.1776938656737.971232085262749810; _gcl_au=1.1.477835274.1776938657; _ga=GA1.1.1468068173.1776938657; _hjSessionUser_386931=eyJpZCI6IjhlZjkxN2FmLTFjNzAtNWI2NS04YTZhLWViNTNmNWMzNzUwMSIsImNyZWF0ZWQiOjE3NzY5Mzg2NTY5MTQsImV4aXN0aW5nIjp0cnVlfQ==; _uetsid=c9dade103efb11f1aac37795cc9017e6; _uetvid=c9daf2003efb11f1b51da7065201e49e; __adroll_fpc=37e02df7b0edda4923a33cc935ec4bcf-1776938657045; _fbp=fb.1.1776938657147.197246373384195974; amplitude_id_0401371089d1a27b189b1976accb81fftrustpilot.com=eyJkZXZpY2VJZCI6ImU0NDdiN2JkLTQ0MzktNGE0Yi1hYTUxLWE0NzZkODg0Mjk4ZCIsInVzZXJJZCI6bnVsbCwib3B0T3V0IjpmYWxzZSwic2Vzc2lvbklkIjoxNzc2OTM4NjU2NzE5LCJsYXN0RXZlbnRUaW1lIjoxNzc2OTM4NjU3NjY3LCJldmVudElkIjoyLCJpZGVudGlmeUlkIjoxLCJzZXF1ZW5jZU51bWJlciI6M30=; amplitude_id_cfe705a69359b8a4c0049d061ee5787btrustpilot.com=eyJkZXZpY2VJZCI6ImU4ZjhmZTIyLTY3NGYtNGYyMC05OTg1LWJkZmZmMzllODU1Y1IiLCJ1c2VySWQiOm51bGwsIm9wdE91dCI6ZmFsc2UsInNlc3Npb25JZCI6MTc3NjkzODYyMDc5NywibGFzdEV2ZW50VGltZSI6MTc3NjkzODcwNDI0MywiZXZlbnRJZCI6OSwiaWRlbnRpZnlJZCI6Miwic2VxdWVuY2VOdW1iZXIiOjExfQ==; TP.uuid=dc58e380-5468-461c-911a-5237c0b1b59a; ajs_anonymous_id=40936936-d108-491a-a9bc-ef2009f320d8; _hjSessionUser_391767=eyJpZCI6IjNjNTE1ODc2LTk1YzctNTI1NS05NjkzLWQ1OGRiOTc1NTdlYyIsImNyZWF0ZWQiOjE3NzY5NDE0MDkyOTQsImV4aXN0aW5nIjp0cnVlfQ==; _hjSession_391767=eyJpZCI6ImYwMzU4MzY3LTQ3NDItNDNhNi04ZDU4LWMxN2ViMjhkZThmZSIsImMiOjE3NzY5NTA4NzQyNDMsInMiOjAsInIiOjAsInNiIjowLCJzciI6MCwic2UiOjAsImZzIjowLCJzcCI6MH0=; tp-consumer-id=637c9877425be3001247489e; ajs_user_id=f8c8f01b18266574ee3aaba612727dcb3ec20b7a; OptanonConsent=isGpcEnabled=0&datestamp=Thu+Apr+23+2026+17%3A39%3A00+GMT%2B0200+(heure+d%E2%80%99%C3%A9t%C3%A9+d%E2%80%99Europe+centrale)&version=202603.1.0&browserGpcFlag=0&isIABGlobal=false&hosts=&consentId=09b25a6a-2e78-4d04-a04f-982f04328961&interactionCount=1&isAnonUser=1&landingPath=NotLandingPage&groups=C0001%3A1%2CC0002%3A1%2CC0003%3A1%2CC0004%3A1&intType=1&geolocation=FR%3BIDF&AwaitingReconsent=false&prevHadToken=0; _ga_11HBWMC274=GS2.1.s1776956838$o5$g1$t1776958741$j59$l0$h0; aws-waf-token=dd14f0c7-7184-4ed9-a2b6-1cb6d151ff5c:DAoAmDdsdlIrAAAA:tKoVz+hOwj1O9s0d/GPeCOU6fyq++ATLLZ4aR3wBz6KJNXlE/+xIYg+JDvTQwO/wREEgzSWZo+ofPV7u1n0+RIziNJTP34HBlvvhabg8KQmdq63EdDaus+V65VNEXLiXeOCjrYR6NyjM2cLrf3eOoWL9M/z9tD4wMWWCnPBh91kXOM6LOdnCtLDODpnjlT0bU+XpoiB3N3AEFZxuRF+gcxjngMd7R8OO6XatmeYOl8W28gUH8sdwQ3eG3YcaDcit0hSBJkzy824=; amplitude_id_67f7b7e6c8cb1b558b0c5bda2f747b07trustpilot.com=eyJkZXZpY2VJZCI6IjQwOTM2OTM2LWQxMDgtNDkxYS1hOWJjLWVmMjAwOWYzMjBkOCIsInVzZXJJZCI6ImY4YzhmMDFiMTgyNjY1NzRlZTNhYWJhNjEyNzI3ZGNiM2VjMjBiN2EiLCJvcHRPdXQiOmZhbHNlLCJzZXNzaW9uSWQiOjE3NzY5NTY4MzM0NjUsImxhc3RFdmVudFRpbWUiOjE3NzY5NTg3NDQ3MjYsImV2ZW50SWQiOjM0LCJpZGVudGlmeUlkIjo5LCJzZXF1ZW5jZU51bWJlciI6NDN9"
    cookie="OptanonAlertBoxClosed=2026-01-06T08:59:54.335Z; jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjb25zdW1lcklkIjoiNjM3Yzk4Nzc0MjViZTMwMDEyNDc0ODllIiwiaGFzQWNjZXB0ZWRUZXJtcyI6dHJ1ZSwiaXNCbG9ja2VkRm9yUmVwb3J0aW5nIjpmYWxzZSwiYWNjZXNzVG9rZW4iOiJCVzF0cmVMYUltQm9CTHBjY2xDdm1COHRCNmdHIiwiYXV0aGVudGljYXRpb25Tb3VyY2UiOiJnb29nbGUiLCJpYXQiOjE3NzA0Mzk4NTksImV4cCI6MTc3ODIxNTg1OX0.mSSYeVlqklfPJidPFTDviljdAX0B7ziyu0n2PpCfEFs; _csrf=PiV8vIH9OBuWfdwWMfaxgfrC; _hjHasCachedUserAttributes=true; analytics_session_id=1776938089235; analytics_session_id.last_access=1776938437995; amplitude_idundefinedtrustpilot.com=eyJvcHRPdXQiOmZhbHNlLCJzZXNzaW9uSWQiOm51bGwsImxhc3RFdmVudFRpbWUiOm51bGwsImV2ZW50SWQiOjAsImlkZW50aWZ5SWQiOjAsInNlcXVlbmNlTnVtYmVyIjowfQ==; _twpid=tw.1776938656737.971232085262749810; _gcl_au=1.1.477835274.1776938657; _ga=GA1.1.1468068173.1776938657; _hjSessionUser_386931=eyJpZCI6IjhlZjkxN2FmLTFjNzAtNWI2NS04YTZhLWViNTNmNWMzNzUwMSIsImNyZWF0ZWQiOjE3NzY5Mzg2NTY5MTQsImV4aXN0aW5nIjp0cnVlfQ==; _uetsid=c9dade103efb11f1aac37795cc9017e6; _uetvid=c9daf2003efb11f1b51da7065201e49e; __adroll_fpc=37e02df7b0edda4923a33cc935ec4bcf-1776938657045; _fbp=fb.1.1776938657147.197246373384195974; amplitude_id_0401371089d1a27b189b1976accb81fftrustpilot.com=eyJkZXZpY2VJZCI6ImU0NDdiN2JkLTQ0MzktNGE0Yi1hYTUxLWE0NzZkODg0Mjk4ZCIsInVzZXJJZCI6bnVsbCwib3B0T3V0IjpmYWxzZSwic2Vzc2lvbklkIjoxNzc2OTM4NjU2NzE5LCJsYXN0RXZlbnRUaW1lIjoxNzc2OTM4NjU3NjY3LCJldmVudElkIjoyLCJpZGVudGlmeUlkIjoxLCJzZXF1ZW5jZU51bWJlciI6M30=; amplitude_id_cfe705a69359b8a4c0049d061ee5787btrustpilot.com=eyJkZXZpY2VJZCI6ImU4ZjhmZTIyLTY3NGYtNGYyMC05OTg1LWJkZmZmMzllODU1Y1IiLCJ1c2VySWQiOm51bGwsIm9wdE91dCI6ZmFsc2UsInNlc3Npb25JZCI6MTc3NjkzODYyMDc5NywibGFzdEV2ZW50VGltZSI6MTc3NjkzODcwNDI0MywiZXZlbnRJZCI6OSwiaWRlbnRpZnlJZCI6Miwic2VxdWVuY2VOdW1iZXIiOjExfQ==; TP.uuid=dc58e380-5468-461c-911a-5237c0b1b59a; ajs_anonymous_id=40936936-d108-491a-a9bc-ef2009f320d8; _hjSessionUser_391767=eyJpZCI6IjNjNTE1ODc2LTk1YzctNTI1NS05NjkzLWQ1OGRiOTc1NTdlYyIsImNyZWF0ZWQiOjE3NzY5NDE0MDkyOTQsImV4aXN0aW5nIjp0cnVlfQ==; _hjSession_391767=eyJpZCI6ImYwMzU4MzY3LTQ3NDItNDNhNi04ZDU4LWMxN2ViMjhkZThmZSIsImMiOjE3NzY5NTA4NzQyNDMsInMiOjAsInIiOjAsInNiIjowLCJzciI6MCwic2UiOjAsImZzIjowLCJzcCI6MH0=; tp-consumer-id=637c9877425be3001247489e; ajs_user_id=f8c8f01b18266574ee3aaba612727dcb3ec20b7a; OptanonConsent=isGpcEnabled=0&datestamp=Thu+Apr+23+2026+17%3A39%3A00+GMT%2B0200+(heure+d%E2%80%99%C3%A9t%C3%A9+d%E2%80%99Europe+centrale)&version=202603.1.0&browserGpcFlag=0&isIABGlobal=false&hosts=&consentId=09b25a6a-2e78-4d04-a04f-982f04328961&interactionCount=1&isAnonUser=1&landingPath=NotLandingPage&groups=C0001%3A1%2CC0002%3A1%2CC0003%3A1%2CC0004%3A1&intType=1&geolocation=FR%3BIDF&AwaitingReconsent=false&prevHadToken=0; _ga_11HBWMC274=GS2.1.s1776956838$o5$g1$t1776958741$j59$l0$h0; aws-waf-token=dd14f0c7-7184-4ed9-a2b6-1cb6d151ff5c:DAoAmDdsdlIrAAAA:tKoVz+hOwj1O9s0d/GPeCOU6fyq++ATLLZ4aR3wBz6KJNXlE/+xIYg+JDvTQwO/wREEgzSWZo+ofPV7u1n0+RIziNJTP34HBlvvhabg8KQmdq63EdDaus+V65VNEXLiXeOCjrYR6NyjM2cLrf3eOoWL9M/z9tD4wMWWCnPBh91kXOM6LOdnCtLDODpnjlT0bU+XpoiB3N3AEFZxuRF+gcxjngMd7R8OO6XatmeYOl8W28gUH8sdwQ3eG3YcaDcit0hSBJkzy824=; amplitude_id_67f7b7e6c8cb1b558b0c5bda2f747b07trustpilot.com=eyJkZXZpY2VJZCI6IjQwOTM2OTM2LWQxMDgtNDkxYS1hOWJjLWVmMjAwOWYzMjBkOCIsInVzZXJJZCI6ImY4YzhmMDFiMTgyNjY1NzRlZTNhYWJhNjEyNzI3ZGNiM2VjMjBiN2EiLCJvcHRPdXQiOmZhbHNlLCJzZXNzaW9uSWQiOjE3NzY5NTY4MzM0NjUsImxhc3RFdmVudFRpbWUiOjE3NzY5NTg3NDQ3MjYsImV2ZW50SWQiOjM0LCJpZGVudGlmeUlkIjo5LCJzZXF1ZW5jZU51bWJlciI6NDN9"
    
    
    cookies = Cookies2python(Cookie)

    headers = {
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'accept-language': 'fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7',
        'cache-control': 'max-age=0',
        'priority': 'u=0, i',
        'referer': 'https://fr.trustpilot.com/review/oscaro.com',
        'sec-ch-ua': '"Google Chrome";v="147", "Not.A/Brand";v="8", "Chromium";v="147"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'document',
        'sec-fetch-mode': 'navigate',
        'sec-fetch-site': 'same-origin',
        'sec-fetch-user': '?1',
        'upgrade-insecure-requests': '1',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36',
        # 'cookie': 'OptanonAlertBoxClosed=2026-01-06T08:59:54.335Z; jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjb25zdW1lcklkIjoiNjM3Yzk4Nzc0MjViZTMwMDEyNDc0ODllIiwiaGFzQWNjZXB0ZWRUZXJtcyI6dHJ1ZSwiaXNCbG9ja2VkRm9yUmVwb3J0aW5nIjpmYWxzZSwiYWNjZXNzVG9rZW4iOiJCVzF0cmVMYUltQm9CTHBjY2xDdm1COHRCNmdHIiwiYXV0aGVudGljYXRpb25Tb3VyY2UiOiJnb29nbGUiLCJpYXQiOjE3NzA0Mzk4NTksImV4cCI6MTc3ODIxNTg1OX0.mSSYeVlqklfPJidPFTDviljdAX0B7ziyu0n2PpCfEFs; _csrf=PiV8vIH9OBuWfdwWMfaxgfrC; _hjHasCachedUserAttributes=true; analytics_session_id=1776938089235; analytics_session_id.last_access=1776938437995; amplitude_idundefinedtrustpilot.com=eyJvcHRPdXQiOmZhbHNlLCJzZXNzaW9uSWQiOm51bGwsImxhc3RFdmVudFRpbWUiOm51bGwsImV2ZW50SWQiOjAsImlkZW50aWZ5SWQiOjAsInNlcXVlbmNlTnVtYmVyIjowfQ==; _twpid=tw.1776938656737.971232085262749810; _gcl_au=1.1.477835274.1776938657; _ga=GA1.1.1468068173.1776938657; _hjSessionUser_386931=eyJpZCI6IjhlZjkxN2FmLTFjNzAtNWI2NS04YTZhLWViNTNmNWMzNzUwMSIsImNyZWF0ZWQiOjE3NzY5Mzg2NTY5MTQsImV4aXN0aW5nIjp0cnVlfQ==; _uetsid=c9dade103efb11f1aac37795cc9017e6; _uetvid=c9daf2003efb11f1b51da7065201e49e; __adroll_fpc=37e02df7b0edda4923a33cc935ec4bcf-1776938657045; _fbp=fb.1.1776938657147.197246373384195974; amplitude_id_0401371089d1a27b189b1976accb81fftrustpilot.com=eyJkZXZpY2VJZCI6ImU0NDdiN2JkLTQ0MzktNGE0Yi1hYTUxLWE0NzZkODg0Mjk4ZCIsInVzZXJJZCI6bnVsbCwib3B0T3V0IjpmYWxzZSwic2Vzc2lvbklkIjoxNzc2OTM4NjU2NzE5LCJsYXN0RXZlbnRUaW1lIjoxNzc2OTM4NjU3NjY3LCJldmVudElkIjoyLCJpZGVudGlmeUlkIjoxLCJzZXF1ZW5jZU51bWJlciI6M30=; amplitude_id_cfe705a69359b8a4c0049d061ee5787btrustpilot.com=eyJkZXZpY2VJZCI6ImU4ZjhmZTIyLTY3NGYtNGYyMC05OTg1LWJkZmZmMzllODU1Y1IiLCJ1c2VySWQiOm51bGwsIm9wdE91dCI6ZmFsc2UsInNlc3Npb25JZCI6MTc3NjkzODYyMDc5NywibGFzdEV2ZW50VGltZSI6MTc3NjkzODcwNDI0MywiZXZlbnRJZCI6OSwiaWRlbnRpZnlJZCI6Miwic2VxdWVuY2VOdW1iZXIiOjExfQ==; TP.uuid=dc58e380-5468-461c-911a-5237c0b1b59a; ajs_anonymous_id=40936936-d108-491a-a9bc-ef2009f320d8; _hjSessionUser_391767=eyJpZCI6IjNjNTE1ODc2LTk1YzctNTI1NS05NjkzLWQ1OGRiOTc1NTdlYyIsImNyZWF0ZWQiOjE3NzY5NDE0MDkyOTQsImV4aXN0aW5nIjp0cnVlfQ==; _hjSession_391767=eyJpZCI6ImYwMzU4MzY3LTQ3NDItNDNhNi04ZDU4LWMxN2ViMjhkZThmZSIsImMiOjE3NzY5NTA4NzQyNDMsInMiOjAsInIiOjAsInNiIjowLCJzciI6MCwic2UiOjAsImZzIjowLCJzcCI6MH0=; tp-consumer-id=637c9877425be3001247489e; ajs_user_id=f8c8f01b18266574ee3aaba612727dcb3ec20b7a; OptanonConsent=isGpcEnabled=0&datestamp=Thu+Apr+23+2026+17%3A19%3A41+GMT%2B0200+(heure+d%E2%80%99%C3%A9t%C3%A9+d%E2%80%99Europe+centrale)&version=202603.1.0&browserGpcFlag=0&isIABGlobal=false&hosts=&consentId=09b25a6a-2e78-4d04-a04f-982f04328961&interactionCount=1&isAnonUser=1&landingPath=NotLandingPage&groups=C0001%3A1%2CC0002%3A1%2CC0003%3A1%2CC0004%3A1&intType=1&geolocation=FR%3BIDF&AwaitingReconsent=false&prevHadToken=0; amplitude_id_67f7b7e6c8cb1b558b0c5bda2f747b07trustpilot.com=eyJkZXZpY2VJZCI6IjQwOTM2OTM2LWQxMDgtNDkxYS1hOWJjLWVmMjAwOWYzMjBkOCIsInVzZXJJZCI6ImY4YzhmMDFiMTgyNjY1NzRlZTNhYWJhNjEyNzI3ZGNiM2VjMjBiN2EiLCJvcHRPdXQiOmZhbHNlLCJzZXNzaW9uSWQiOjE3NzY5NTY4MzM0NjUsImxhc3RFdmVudFRpbWUiOjE3NzY5NTc1ODEyODYsImV2ZW50SWQiOjMwLCJpZGVudGlmeUlkIjo3LCJzZXF1ZW5jZU51bWJlciI6Mzd9; _ga_11HBWMC274=GS2.1.s1776956838$o5$g1$t1776958046$j60$l0$h0; aws-waf-token=dd14f0c7-7184-4ed9-a2b6-1cb6d151ff5c:DAoAoGdr3W0ZAAAA:yr/3fLb14ZrXmfxV8bNTmItDJcTS90ll3EuwQzaqcO/3/ZalhcJ31V5M3rBB8ADPHiOX+d9rxt87g48cxInWx09UMns9d5krjpVSR+HQHtC4oqbryeNX9DmBb9SlQVp8XK8uu/CTZMbzqDChGEK5OqQwMNcLRl5VMHb1WDV/fmfRRqydn8AUi2qoYm2jsxfFVRatgD46JzzJ6hZrNY2ZusemG58tAznSo49fn4ATkYa9sKw6O3TPUCwfizNjnDHYrVfS5zUVPbM=',
    }

    params = {
        'page': '2',
    }

    response = requests.get('https://fr.trustpilot.com/review/oscaro.com', cookies=cookies, headers=headers)

    #response = requests.get("https://fr.trustpilot.com/review/oscaro.com", headers=headers, cookies=cookies)

    # 2. Extraire le buildId depuis __NEXT_DATA__

    soup = BeautifulSoup(response.text, "html.parser")
    display(HTML(soup))

    found=soup.find("script", {"id": "__NEXT_DATA__"})
    print(found)
    try:
        next_data = json.loads(found.string)

        build_id = next_data["buildId"]
        print(build_id)  # ex: businessunitprofile-consumersite-2.6055.0

        # 3. Construire l'URL JSON proprement
        json_url = f"https://fr.trustpilot.com/_next/data/{build_id}/review/oscaro.com.json?businessunit=oscaro.com"

        # 4. Appeler l'API JSON directement
        data = requests.get(json_url, headers=headers, cookies=cookies).json()
        reviews = data["pageProps"]["reviews"]

        for r in reviews:
            print(r["consumer"]["displayName"], "⭐", r["rating"])
            print(r["text"][:150])
    except Exception as e:
        print(f"💥An exception occurred:{e}")


######################################################################
####### only preparing local data fetch with proper calendar


import pandas as pd
from pathlib import Path
import csv
import datetime as dt
import yaml

def get_calendar():
    #this calendar is providing date with hours depending on the time we run the function - so bad
    #calendar = pd.DataFrame({"date":[datetime.now() - timedelta(days=p-3650) for p in range(19800)]})
    # we compute the delta + we project to first 0:00 of the day
    Today=dt.datetime.combine(dt.datetime.today(), dt.datetime.min.time())

    calendar = pd.DataFrame({"date":[Today - dt.timedelta(days=p-3650) for p in range(19800)]})
    calendar["Period"] ='P'+calendar['date'].dt.isocalendar().year.astype(str) + \
        calendar['date'].dt.isocalendar().week.astype(str).str.zfill(2)   
    return calendar

calend= get_calendar()
#various ways to query a calendar
lst53s=calend[ calend.Period.str.endswith("53") ]
#from day to day get the periods

def get_periods(StartDay=None,EndDay=None,NbPer=None):
    #logic d'usage
    #début fin
    #début + N
    #fin - N
    #"1998-01-01","1998-12-31"
    calend=get_calendar()

    if NbPer != None:        
        NbDays=dt.timedelta(days=7*(NbPer-1))
        if StartDay == None:
            StartDay=dt.datetime.strptime(EndDay, '%Y-%m-%d')-NbDays 
        if EndDay == None:
            EndDay=dt.datetime.strptime(StartDay, '%Y-%m-%d')-NbDays 
    else:
        if StartDay == None or EndDay == None:
            raise "inconsistent settings"
    lstPer=calend[ calend.date.between(StartDay,EndDay) ].Period.sort_values(axis=0).unique()
    #lstPer=calend[ calend.date.between("1998-01-01","1998-12-31") ].Period.sort_values(axis=0).unique()
    return lstPer

#ASSEMBLAGE AVIS UNIQUE
def get_avis(avisSrc):
    avisF = Path(f"../data/{avisSrc}.csv")
    if avisF.exists():
        return pd.read_csv(avisF,sep=";")

def merge_avis(set1="avis_40k",set2="avis_28k", tgtset="avis_45k"):
    df=pd.concat([get_avis(set1),get_avis(set2)])

    nodupkeys=["auteur","note","titre","commentaire","date_experience","texteServiceReply","dateServiceReply","dataServiceReply"]
    df = df.drop(columns=["url"])
    df = df.sort_values(by=nodupkeys)
    df = df.drop_duplicates(subset=nodupkeys)
    #display(df.head())

    df.info(memory_usage="deep")
    tgtAvis=Path(f"../data/{tgtset}.csv")
    df.to_csv(tgtAvis, index=None,sep=";",quoting=csv.QUOTE_ALL)

def get_reviews(reviewSrc="avis_45k", arrPeriods=[], mode="bulk"):
    """ fonction API de review 
    fournit le jeux de review disponible par filtre par periode
    on fournit deux périodes
    
    mode peut prendre les valeurs "bulk" on "incremental"

    cela détermine uniquement le jeux du csv de sortie
    -d collect.py
    -P params.yaml
    -o data/reviews.csv 
    -o data/reviews_inc.csv
    python collect.py
    """

    df=get_avis(reviewSrc)

    df['date_experience']  = pd.to_datetime(df['date_experience'], errors='coerce')
    df['date_avis']        = pd.to_datetime(df['date_avis'], errors='coerce')
    df['dateServiceReply'] = pd.to_datetime(df['dateServiceReply'], errors='coerce')

    df['Period']= 'P'+df['date_experience'].dt.isocalendar().year.astype(str) + \
        df['date_experience'].dt.isocalendar().week.astype(str).str.zfill(2)

    df['YearMonth_experience']  = df['date_experience'].dt.to_period('M')
    df['YearMonth_avis']        = df['date_avis'].dt.to_period('M')
    df['YearMonthServiceReply'] = df['dateServiceReply'].dt.to_period('M')
    if len(arrPeriods) == 0:
        return df
    else:
        if len(arrPeriods) <= 2:
            return df[ df.Period.between(arrPeriods[0],arrPeriods[-1]) ]
        else:
            return df[ df.Period.isin(arrPeriods) ]
        
#main section for dvc runs with the params.yaml
if __name__ == "__main__":

    #from params.yaml
    
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    mode = params["collect"]["mode"]
    source = params["collect"]["source"]
    period_start = params["collect"]["period_start"]
    period_end = params["collect"]["period_end"]

    # collecte le batch demandé
    df_new = get_reviews(reviewSrc=source,period=[period_start, period_end], mode=mode)

    if mode == "incremental":
        # append sur le fichier existant
        if os.path.exists("data/raw/reviews.csv"):
            df_existing = pd.read_csv("data/raw/reviews.csv")
            df = pd.concat([df_existing, df_new]).drop_duplicates()
        else:
            df = df_new
        df.to_csv("data/raw/reviews.csv", index=False)

    elif mode == "full":
        # écrase tout
        df = df_new
        df.to_csv("data/raw/reviews.csv", index=False)

    elif mode == "predict":
        # on prépare un batch de données pour la prédiction
        df = df_new
        df.to_csv("data/raw/reviews2predict.csv", index=False)

