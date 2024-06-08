import requests
from bs4 import BeautifulSoup as BS
from tqdm import tqdm

def get_lyrics_link():
    links = []
    for i in tqdm(range(1, 199)):
        url= f"http://liriklagu.net/melayu/page/{i}"
        response = requests.get(url)
        bs = BS(response.text, "lxml")
        all_h2 = bs.find_all("h2", class_="post-title")
        for h2 in all_h2:
            href = h2.find("a", href=True)["href"]
            links.append(href)
    return links

def get_data(links:list, start, end):
    full_data = {"title":[], "body":[]}
    
    for link in tqdm(links[start:end]):
        try:
            response = requests.get(link)
            tajuk = link.split("/")[-2].replace("-", " ").title()
            penyanyi = link.split("/")[-3].replace("-", " ").title()
            title = f"{tajuk} - {penyanyi}"
            bs = BS((response.text).replace("<br>", "\n"), "lxml")
            div = bs.find("div", class_="post-content entry-content")
            p_tag = div.find_all("p")
    
            lyrics = []
            for p in p_tag:
                text = p.text
                lyrics.append(text)
    
            full_text = "\n".join(lyrics)
            full_data["title"].append(title)
            full_data["body"].append(full_text)
        except:
            print("Error", link)
            continue
            
    return full_data

def save_to_txt(full_data, file_name):
    with open(file_name, 'w', encoding='utf-8') as f:
        for title, body in zip(full_data['title'], full_data['body']):
            f.write(f"{title}\n{body}\n\n{'-'*50}\n\n")

if __name__ == "__main__":
    links = get_lyrics_link()
    full_data = get_data(links, 0, None) # index 0 till end
    save_to_txt(full_data, "lirik_lagu.txt")
