# 🏈 FF-Scraping-and-Visualizations

## What does this project do?
- Scrapes entire fantasy league history from NFL.com.  
- Exports all standings and games as CSV files in `./output`.  
- Aggregates standings into a single CSV file.  
- Iterates through all games to find biggest blowouts and narrowest victories.  
- 📊 **(New)** Includes a visualization notebook (`/output/Visualizations.ipynb`) to highlight long-term trends and rivalries.  

---

## How to run this:
```bash
git clone https://github.com/CyberJrod/FF-Scraping-and-Visualization
```

1. In `constants.py`, update with your league ID and start/end years.  
2. In `cookieString.py`, update cookie string with an active NFL.com cookie.  
   - You can find this by inspecting a request in Chrome DevTools.  
3. Run the scrapers:  
   - `python scrapeStandings.py` → scrapes all standings.  
   - `python aggregateStandings.py` → aggregates into 1 CSV.  
   - `python scrapeGamecenter.py` → scrapes all games.  
   - `python analyzeGamecenter.py` → finds biggest blowouts and narrowest margins of victory.  
4. 📊 Open `/output/Visualizations.ipynb` to generate charts and tables.  

---

## 📊 Visualizations in `Visualizations.ipynb`
The notebook processes scraped data and produces league insights such as:  

- **Championships & 1st Losers**  
  - Who has the most titles and runner-up finishes.  
  - Finals summary with regular season record and points.  

- **Top 3 Finishes**  
  - Number of top-3 playoff finishes vs. seasons played.  

- **Bottom Finishes (Last Place & Bottom 3)**  
  - Managers with the most struggles.  
  - Ratios of bad finishes to total seasons played.  

- **Head-to-Head Records**  
  - Win/loss matrix between managers.  
  - Heatmap visualization of rivalries across seasons.  

---

## Known Issues
- If multiple team managers have the same name, their results will be aggregated together.  
- The script assumes top half of the league makes playoffs.  

---

## Attribution
- Original scraper: [FF-Scraping](https://github.com/PeteTheHeat/FF-Scraping) by **@PeteTheHeat**.  
- Visualization notebook: added in this fork [FF-Scraping-and-Visualization](https://github.com/CyberJrod/FF-Scraping-and-Visualization) by **@CyberJrod**.  
