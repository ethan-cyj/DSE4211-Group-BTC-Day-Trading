This readme outlines what kind of data is in the 4 loose files,
1. daily_active_addresses.csv
2. miniwhale_movements.csv
3. googletrends_interest_2025-03-09.csv
4. GDELTS 9 March Pull.xlsx

####################################################

daily_active_addresses.csv
--------------------------
	       day		     |	daily_active_addresses  |	buy_ins         |	sells     |
              ----		     |		----		|	----	        |	----	  |	
0	2024-01-01 00:00:00.000 UTC  | 	       828389		|	624627	        |	560256    |
1	2024-01-02 00:00:00.000 UTC  |	       866366		|	550845  	|	571551    |
2	2024-01-03 00:00:00.000 UTC  |	       947432		| 	677057  	|	630795    |


Source: This is on-chain data. Meaning insights that you can attain from the data stored on the blockchain. For this, I used Dune, but can be found using apps like etherscan or hosting a block, etc.

day is the day 
daily_active_addresses is the number of wallet addresses that either buy or sold bitcoin
buy_ins is the number of wallet addresses that bought bitcoin
sells is the number of wallet addresses that sold bitcoin

Just in case anyone wondering, buy_ins + sells != daily_active_addresses because a transaction sometimes consists of one person sell, one person buy. 

####################################################

miniwhale_movements.csv
--------------------------
	day			     |  total_in_value_btc   |	total_out_value_btc    |	tx_id   |
              ----		     |    ----		     |   	----	       |        ----    |
0	2024-01-01 00:00:00.000 UTC  |	143.8841485	     |	143.8818236	       |  0x001c863ed706685752b1b166d1100d78bac93ed0d1c868edca53dea8177fbfc3    |
1	2024-01-01 00:00:00.000 UTC  |	200.6185094	     |  200.6173321	       |  0x004238d08918b7b9f4d1d4e0470e3fd891e8529dee175c89d2b88852f96cbf31  |
2	2024-01-01 00:00:00.000 UTC  |	269.6588517	     |  269.6584033	       |  0x007c1e971e1e286e82232cb0336829d68719eb6fc8f865cc862cd9b932984ddb  |



Source: Also on-chain data. For this, I used Dune, but can be found using apps like etherscan or hosting a block, etc. 

Whales are people like a LOT of bitcoin
Whale transaction is usually 1000+ transacted bitcoins. Here, I just used transaction of 100 bitcoins, so i called it miniwhale! 100 btc is still $8,500,000

total_in_value_btc and total_out_value_btc differs by a little due to transaction fees


####################################################

googletrends_interest_2025-03-09.csv
--------------------------
	searchTerm	weekLabel	interestScore
0	crypto	Mar 3â€‰â€“â€‰9, 2024	70
1	crypto	Mar 10â€‰â€“â€‰16, 2024	65
2	crypto	Mar 17â€‰â€“â€‰23, 2024	53


Source: Google trends data sourced from apify because pytrends doesn't work anymore 😭

Dates format a bit weird due to exporting as csv but can still tell HAHHAHA

Search term consists of KEYWORDS = ["cryptocurrency", "crypto", "bitcoin", "BTC", "ethereum", "eth"]

Interest scores are normalised with respect to highest and lowest point in the range of the year


####################################################

GDELTS 9 March Pull.xlsx
--------------------------
event_date	avg_sentiment	avg_positive_score	avg_negative_score	avg_polarity	article_count	sample_articles	key_persons	key_organizations	key_locations


Source: GDELTSv2 which is a relational database of all news to ever exist pulled from bigquery. I filtered for 
      LOWER(Themes) LIKE '%blockchain%'
      OR LOWER(Themes) LIKE '%bitcoin%'
      OR LOWER(Themes) LIKE '%cryptocurrency%'
https://docs.google.com/spreadsheets/d/1K_zYrS3gqyLMOl4WwYWGNWHHxlTxn-fYv8dI0z_FRNI/edit?usp=sharing

Above I only showed the column titles because some strings are damn long and messy.

event_date		is the date
avg_sentiment		average OVERALL sentiment of all the news that day
avg_positive_score	average POSITIVE sentiment of all the news that day
avg_negative_score	average NEGATIVE sentiment of all the news that day
avg_polarity		average POLARITY (difference between positive and negative) sentiment of all the news that day
article_count		is the count of all articles that day
sample_articles	    	is 5 sample articles (the weblink) in case we wanna check them out
key_persons		is all the people involved (eg. Article on Trump's precidency on bitcoin will have Trump here)
key_organizations	is all the organisations involved
key_locations		is all the locations involved



