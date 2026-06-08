from kgpipe_llm.any_extraction import extract_ontology_surface_triples
from pathlib import Path
from kgpipe_llm.common.core import LLMClient
import os

TEXT="""
Titanic is a 1997 American epic historical romance film written and directed by James Cameron. Incorporating both historical and fictional aspects, it is based on accounts of the sinking of RMS Titanic in 1912. Leonardo DiCaprio and Kate Winslet star as members of different social classes who fall in love during the ship's ill-fated maiden voyage. The ensemble cast includes Billy Zane, Kathy Bates, Frances Fisher, Bernard Hill, Jonathan Hyde, Danny Nucci, David Warner and Bill Paxton. Cameron's inspiration came from his fascination with shipwrecks. He felt a love story interspersed with human loss would be essential to convey the emotional impact of the disaster. Production began on September 1, 1995, when Cameron shot footage of the Titanic wreck. The modern scenes were shot on board the Shirshov Institute of Oceanology research vessel Akademik Mstislav Keldysh, which Cameron had used as a base when filming the wreck. Scale models, computer-generated imagery (CGI), and a reconstruction of the Titanic built at Baja Studios were used to recreate the sinking. Titanic was initially in development at 20th Century Fox, but delays and a mounting budget resulted in Fox partnering with Paramount Pictures for financial help. It was the most expensive film ever made at the time, with a production budget of $200 million. Filming took place from July 1996 to March 1997. Titanic premiered at the Tokyo International Film Festival on November 1, 1997, and was released in the United States on December 19. It was distributed by Paramount Pictures in the United States and Canada and by 20th Century Fox in other territories. It was praised for its visual effects, performances (particularly those of DiCaprio, Winslet, and Gloria Stuart), production values, direction, score, cinematography, story, and emotional depth. Among other awards, the film received fourteen nominations at the 70th Academy Awards and won eleven, including Best Picture and Best Director. In doing so, it tied both All About Eve (1950) for the record for the most Academy Award nominations, and Ben-Hur (1959) for the most Academy Awards won by a film, making Titanic the most successful individual film in Academy Award history (these records would be matched by 2016's La La Land and 2003's The Lord of the Rings: The Return of the King respectively, although the nomination record was surpassed by 2025's Sinners in 2026). With an initial worldwide gross of over $1.84 billion, Titanic was the first film to reach the billion-dollar mark (1993's Jurassic Park would later become the earliest-released film to achieve this feat, via subsequent re-releases), and was the highest-grossing film of all time until Cameron's next film, Avatar (2009), surpassed it in 2010. Income from the initial theatrical release, retail video, and soundtrack sales and US broadcast rights exceeded $3.2 billion. Releases pushed the worldwide theatrical total to $2.264 billion, making Titanic the second film to gross more than $2 billion worldwide after Avatar; as of 2023, it is the fourth-highest-grossing film. In 2017, the Library of Congress selected it for preservation in the United States National Film Registry as "culturally, historically, or aesthetically significant
"""

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY is not set")

model_name="o4-mini"
ontology_path = Path("/home/marvin/phd/data/moviekg/datasets/film_10k/ontology.ttl")

def test_extract_ontology_surface_triples():
    client = LLMClient(
        model_name=model_name,
        token=API_KEY,
        api_type="openai",
    )
    result = extract_ontology_surface_triples(TEXT, ontology_path, client)
    print(result.model_dump_json(indent=2))