"""
High-Fidelity Multi-Domain Synthetic Data Generator for Omni-RecSys
Simulates realistic interaction patterns, user personas, item catalogs, and artwork variants.
Provides 100+ real, iconic movies, songs, products, and short clips with diverse, verified imagery.
"""

import random
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any


class SyntheticDataGenerator:
    """
    Generates rich, realistic datasets for all 4 supported domains:
    - Movies / Video (Netflix style with real iconic titles & artwork variants)
    - Music (Spotify style with acoustic attributes & real track titles)
    - E-Commerce (Amazon style with realistic brands & gear)
    - Short Feed (TikTok style with viral creator concepts & sound IDs)
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)

        # Verified, high-resolution cinematic and thematic image pools
        self.scifi_images = [
            "https://images.unsplash.com/photo-1451187580459-43490279c0fa?w=600&q=80",
            "https://images.unsplash.com/photo-1518709268805-4e9042af9f23?w=600&q=80",
            "https://images.unsplash.com/photo-1542751371-adc38448a05e?w=600&q=80",
            "https://images.unsplash.com/photo-1526374965328-7f61d4dc18c5?w=600&q=80",
            "https://images.unsplash.com/photo-1509316975850-ff9c5deb0cd9?w=600&q=80",
            "https://images.unsplash.com/photo-1520072959219-c595dc870360?w=600&q=80",
            "https://images.unsplash.com/photo-1506703719100-a0f3a48c0f86?w=600&q=80",
            "https://images.unsplash.com/photo-1446776811953-b23d57bd21aa?w=600&q=80",
            "https://images.unsplash.com/photo-1518770660439-4636190af475?w=600&q=80",
            "https://images.unsplash.com/photo-1508997449629-303059a039c0?w=600&q=80",
        ]

        self.action_images = [
            "https://images.unsplash.com/photo-1509198397868-475647b2a1e5?w=600&q=80",
            "https://images.unsplash.com/photo-1568605117036-5fe5e7bab0b7?w=600&q=80",
            "https://images.unsplash.com/photo-1536240478700-b869070f9279?w=600&q=80",
            "https://images.unsplash.com/photo-1552519507-da3b142c6e3d?w=600&q=80",
            "https://images.unsplash.com/photo-1534447677768-be436bb09401?w=600&q=80",
            "https://images.unsplash.com/photo-1541701494587-cb58502866ab?w=600&q=80",
            "https://images.unsplash.com/photo-1519074069444-1ba4ea16e60b?w=600&q=80",
            "https://images.unsplash.com/photo-1563245372-f21724e3856d?w=600&q=80",
            "https://images.unsplash.com/photo-1517457373958-b7bdd4587205?w=600&q=80",
            "https://images.unsplash.com/photo-1533488765986-dfa2a9939acd?w=600&q=80",
        ]

        self.crime_thriller_images = [
            "https://images.unsplash.com/photo-1594909122845-11baa439b7bf?w=600&q=80",
            "https://images.unsplash.com/photo-1478760329108-5c3ed9d495a0?w=600&q=80",
            "https://images.unsplash.com/photo-1514565131-fce0801e5785?w=600&q=80",
            "https://images.unsplash.com/photo-1485846234645-a62644f84728?w=600&q=80",
            "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=600&q=80",
            "https://images.unsplash.com/photo-1574267432553-4b4628081c31?w=600&q=80",
            "https://images.unsplash.com/photo-1492691527719-9d1e07e534b4?w=600&q=80",
            "https://images.unsplash.com/photo-1506744038136-46273834b3fb?w=600&q=80",
        ]

        self.romance_drama_images = [
            "https://images.unsplash.com/photo-1516589178581-6cd7833ae3b2?w=600&q=80",
            "https://images.unsplash.com/photo-1518199266791-5375a83190b7?w=600&q=80",
            "https://images.unsplash.com/photo-1494774157365-9e04c6720e47?w=600&q=80",
            "https://images.unsplash.com/photo-1515934751635-c81c6bc9a2d8?w=600&q=80",
            "https://images.unsplash.com/photo-1529333166437-7750a6dd5a70?w=600&q=80",
            "https://images.unsplash.com/photo-1511285560929-80b456fea0bc?w=600&q=80",
            "https://images.unsplash.com/photo-1474552226712-ac0f0961a954?w=600&q=80",
            "https://images.unsplash.com/photo-1516575334481-f85287c2c82d?w=600&q=80",
            "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=600&q=80",
        ]

        self.animation_images = [
            "https://images.unsplash.com/photo-1578632767115-351597cf2477?w=600&q=80",
            "https://images.unsplash.com/photo-1579783902614-a3fb3927b675?w=600&q=80",
            "https://images.unsplash.com/photo-1579783900882-c0d3dad7b119?w=600&q=80",
            "https://images.unsplash.com/photo-1563089145-599997674d42?w=600&q=80",
            "https://images.unsplash.com/photo-1550684848-fac1c5b4e853?w=600&q=80",
            "https://images.unsplash.com/photo-1579546929518-9e396f3cc809?w=600&q=80",
            "https://images.unsplash.com/photo-1550684847-75bdda21cc95?w=600&q=80",
            "https://images.unsplash.com/photo-1519638399535-1b036603ac77?w=600&q=80",
        ]

        self.music_images = [
            "https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?w=600&q=80",
            "https://images.unsplash.com/photo-1470225620780-dba8ba36b745?w=600&q=80",
            "https://images.unsplash.com/photo-1508700115892-45ecd05ae2ad?w=600&q=80",
            "https://images.unsplash.com/photo-1516450360452-9312f5e86fc7?w=600&q=80",
            "https://images.unsplash.com/photo-1460723237483-7a6dc9d0b212?w=600&q=80",
            "https://images.unsplash.com/photo-1506157786151-b8491531f063?w=600&q=80",
            "https://images.unsplash.com/photo-1513151233558-d860c5398176?w=600&q=80",
            "https://images.unsplash.com/photo-1511379938547-c1f69419868d?w=600&q=80",
            "https://images.unsplash.com/photo-1445985543470-41fba5c3144a?w=600&q=80",
            "https://images.unsplash.com/photo-1493225457124-a3eb161ffa5f?w=600&q=80",
        ]

        self.ecom_images = [
            "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=600&q=80",
            "https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=600&q=80",
            "https://images.unsplash.com/photo-1546868871-7041f2a55e12?w=600&q=80",
            "https://images.unsplash.com/photo-1583394838336-acd977736f90?w=600&q=80",
            "https://images.unsplash.com/photo-1527443224154-c4a3942d3acf?w=600&q=80",
            "https://images.unsplash.com/photo-1572536147248-ac59a8abfa4b?w=600&q=80",
            "https://images.unsplash.com/photo-1511707171634-5f897ff02aa9?w=600&q=80",
            "https://images.unsplash.com/photo-1585060544812-6b45742d762f?w=600&q=80",
            "https://images.unsplash.com/photo-1593642632823-8f785ba67e45?w=600&q=80",
        ]

        self.short_images = [
            "https://images.unsplash.com/photo-1611162617213-7d7a39e9b1d7?w=600&q=80",
            "https://images.unsplash.com/photo-1534447677768-be436bb09401?w=600&q=80",
            "https://images.unsplash.com/photo-1517457373958-b7bdd4587205?w=600&q=80",
            "https://images.unsplash.com/photo-1550684848-fac1c5b4e853?w=600&q=80",
            "https://images.unsplash.com/photo-1518709268805-4e9042af9f23?w=600&q=80",
            "https://images.unsplash.com/photo-1516450360452-9312f5e86fc7?w=600&q=80",
            "https://images.unsplash.com/photo-1536240478700-b869070f9279?w=600&q=80",
            "https://images.unsplash.com/photo-1513151233558-d860c5398176?w=600&q=80",
        ]

    def _get_image_for_genre(self, genre: str, idx: int) -> str:
        g = genre.lower()
        if "sci-fi" in g or "futur" in g or "tech" in g:
            return self.scifi_images[idx % len(self.scifi_images)]
        elif "action" in g or "war" in g or "superhero" in g:
            return self.action_images[idx % len(self.action_images)]
        elif "crime" in g or "thriller" in g or "noir" in g or "mystery" in g:
            return self.crime_thriller_images[idx % len(self.crime_thriller_images)]
        elif "romance" in g or "drama" in g:
            return self.romance_drama_images[idx % len(self.romance_drama_images)]
        elif "animation" in g or "anime" in g or "fantasy" in g:
            return self.animation_images[idx % len(self.animation_images)]
        else:
            all_imgs = self.scifi_images + self.action_images + self.romance_drama_images
            return all_imgs[idx % len(all_imgs)]

    # -------------------------------------------------------------
    # 1. MOVIES CATALOG (Netflix Style)
    # -------------------------------------------------------------
    def generate_movies_catalog(self, num_items: int = 130) -> pd.DataFrame:
        famous_movies = [
            ("Inception", "Sci-Fi", 2010, 8.8, "Christopher Nolan"),
            ("Interstellar", "Sci-Fi", 2014, 8.7, "Christopher Nolan"),
            ("The Dark Knight", "Action", 2008, 9.0, "Christopher Nolan"),
            ("Pulp Fiction", "Crime", 1994, 8.9, "Quentin Tarantino"),
            ("Spirited Away", "Animation", 2001, 8.6, "Hayao Miyazaki"),
            ("Parasite", "Thriller", 2019, 8.5, "Bong Joon-ho"),
            ("Dune: Part Two", "Sci-Fi", 2024, 8.7, "Denis Villeneuve"),
            ("Oppenheimer", "Drama", 2023, 8.9, "Christopher Nolan"),
            ("The Matrix", "Sci-Fi", 1999, 8.7, "The Wachowskis"),
            ("Fight Club", "Drama", 1999, 8.8, "David Fincher"),
            ("Goodfellas", "Crime", 1990, 8.7, "Martin Scorsese"),
            ("The Shawshank Redemption", "Drama", 1994, 9.3, "Frank Darabont"),
            ("Whiplash", "Drama", 2014, 8.5, "Damien Chazelle"),
            ("Blade Runner 2049", "Sci-Fi", 2017, 8.0, "Denis Villeneuve"),
            ("Spider-Man: Across the Spider-Verse", "Animation", 2023, 8.7, "Joaquim Dos Santos"),
            ("Everything Everywhere All at Once", "Sci-Fi", 2022, 7.8, "Daniel Kwan"),
            ("Stranger Things: Season 4", "Sci-Fi", 2022, 8.7, "The Duffer Brothers"),
            ("Squid Game", "Thriller", 2021, 8.0, "Hwang Dong-hyuk"),
            ("Cyberpunk: Edgerunners", "Animation", 2022, 8.3, "Hiroyuki Imaishi"),
            ("Severance", "Thriller", 2022, 8.7, "Ben Stiller"),
            ("The Godfather", "Crime", 1972, 9.2, "Francis Ford Coppola"),
            ("Gladiator", "Action", 2000, 8.5, "Ridley Scott"),
            ("Alien", "Horror", 1979, 8.5, "Ridley Scott"),
            ("Jurassic Park", "Sci-Fi", 1993, 8.2, "Steven Spielberg"),
            ("Forrest Gump", "Drama", 1994, 8.8, "Robert Zemeckis"),
            ("The Prestige", "Mystery", 2006, 8.5, "Christopher Nolan"),
            ("Shutter Island", "Thriller", 2010, 8.2, "Martin Scorsese"),
            ("Joker", "Crime", 2019, 8.4, "Todd Phillips"),
            ("Avengers: Endgame", "Action", 2019, 8.4, "Anthony & Joe Russo"),
            ("Star Wars: The Empire Strikes Back", "Sci-Fi", 1980, 8.7, "Irvin Kershner"),
            ("The Lord of the Rings: Return of the King", "Fantasy", 2003, 9.0, "Peter Jackson"),
            ("Back to the Future", "Sci-Fi", 1985, 8.5, "Robert Zemeckis"),
            ("Mad Max: Fury Road", "Action", 2015, 8.1, "George Miller"),
            ("Akira", "Animation", 1988, 8.0, "Katsuhiro Otomo"),
            ("Your Name", "Animation", 2016, 8.4, "Makoto Shinkai"),
            ("Princess Mononoke", "Animation", 1997, 8.4, "Hayao Miyazaki"),
            ("Taxi Driver", "Crime", 1976, 8.2, "Martin Scorsese"),
            ("Se7en", "Crime", 1995, 8.6, "David Fincher"),
            ("The Silence of the Lambs", "Thriller", 1991, 8.6, "Jonathan Demme"),
            ("No Country for Old Men", "Crime", 2007, 8.2, "Ethan & Joel Coen"),
            ("Django Unchained", "Western", 2012, 8.5, "Quentin Tarantino"),
            ("Inglourious Basterds", "War", 2009, 8.4, "Quentin Tarantino"),
            ("The Wolf of Wall Street", "Comedy", 2013, 8.2, "Martin Scorsese"),
            ("Titanic", "Romance", 1997, 7.9, "James Cameron"),
            ("La La Land", "Romance", 2016, 8.0, "Damien Chazelle"),
            ("Her", "Romance", 2013, 8.0, "Spike Jonze"),
            ("Eternal Sunshine of the Spotless Mind", "Romance", 2004, 8.3, "Michel Gondry"),
            ("Before Sunrise", "Romance", 1995, 8.1, "Richard Linklater"),
            ("Pride & Prejudice", "Romance", 2005, 7.8, "Joe Wright"),
            ("About Time", "Romance", 2013, 7.8, "Richard Curtis"),
            ("Coco", "Animation", 2017, 8.4, "Lee Unkrich"),
            ("Toy Story", "Animation", 1995, 8.3, "John Lasseter"),
            ("WALL-E", "Animation", 2008, 8.4, "Andrew Stanton"),
            ("Up", "Animation", 2009, 8.3, "Pete Docter"),
            ("Ratatouille", "Animation", 2007, 8.1, "Brad Bird"),
            ("The Lion King", "Animation", 1994, 8.5, "Roger Allers"),
            ("Top Gun: Maverick", "Action", 2022, 8.3, "Joseph Kosinski"),
            ("John Wick: Chapter 4", "Action", 2023, 7.7, "Chad Stahelski"),
            ("Casino Royale", "Action", 2006, 8.0, "Martin Campbell"),
            ("Mission: Impossible - Fallout", "Action", 2018, 7.7, "Christopher McQuarrie"),
            ("The Departed", "Crime", 2006, 8.5, "Martin Scorsese"),
            ("Arrival", "Sci-Fi", 2016, 7.9, "Denis Villeneuve"),
            ("Drive", "Action", 2011, 7.8, "Nicolas Winding Refn"),
            ("Nightcrawler", "Thriller", 2014, 7.8, "Dan Gilroy"),
            ("The Grand Budapest Hotel", "Comedy", 2014, 8.1, "Wes Anderson"),
            ("The Social Network", "Drama", 2010, 7.8, "David Fincher"),
            ("Knives Out", "Mystery", 2019, 7.9, "Rian Johnson"),
            ("1917", "War", 2019, 8.2, "Sam Mendes"),
            ("Ford v Ferrari", "Drama", 2019, 8.1, "James Mangold"),
            ("Get Out", "Horror", 2017, 7.8, "Jordan Peele"),
            ("Hereditary", "Horror", 2018, 7.3, "Ari Aster"),
            ("A Quiet Place", "Horror", 2018, 7.5, "John Krasinski"),
            ("The Shining", "Horror", 1980, 8.4, "Stanley Kubrick"),
            ("Black Panther", "Action", 2018, 7.3, "Ryan Coogler"),
            ("Iron Man", "Action", 2008, 7.9, "Jon Favreau"),
            ("Guardians of the Galaxy", "Sci-Fi", 2014, 8.0, "James Gunn"),
            ("Dune: Part One", "Sci-Fi", 2021, 8.0, "Denis Villeneuve"),
            ("Chernobyl", "Drama", 2019, 9.3, "Craig Mazin"),
            ("The Last of Us", "Drama", 2023, 8.8, "Craig Mazin"),
            ("Succession", "Drama", 2023, 8.9, "Jesse Armstrong"),
            ("Breaking Bad", "Crime", 2013, 9.5, "Vince Gilligan"),
            ("Better Call Saul", "Crime", 2022, 9.0, "Peter Gould"),
            ("Game of Thrones", "Fantasy", 2019, 9.2, "David Benioff"),
            ("House of the Dragon", "Fantasy", 2024, 8.4, "Ryan J. Condal"),
            ("Arcane: League of Legends", "Animation", 2021, 9.0, "Christian Linke"),
            ("Ted Lasso", "Comedy", 2023, 8.8, "Brendan Hunt"),
            ("The Bear", "Drama", 2023, 8.6, "Christopher Storer"),
            ("Peaky Blinders", "Crime", 2022, 8.8, "Steven Knight"),
            ("Fargo", "Crime", 2024, 8.9, "Noah Hawley"),
            ("True Detective", "Crime", 2014, 8.9, "Nic Pizzolatto"),
            ("Mindhunter", "Crime", 2019, 8.6, "Joe Penhall"),
            ("Narcos", "Crime", 2017, 8.8, "Chris Brancato"),
            ("Dark", "Sci-Fi", 2020, 8.7, "Baran bo Odar"),
            ("Black Mirror", "Sci-Fi", 2023, 8.7, "Charlie Brooker"),
            ("The Queen's Gambit", "Drama", 2020, 8.5, "Scott Frank"),
            ("The Boys", "Action", 2024, 8.7, "Eric Kripke"),
            ("Invincible", "Animation", 2023, 8.7, "Robert Kirkman"),
            ("Beef", "Comedy", 2023, 8.0, "Lee Sung Jin"),
            ("The Mandalorian", "Sci-Fi", 2023, 8.6, "Jon Favreau"),
            ("The Batman", "Action", 2022, 7.8, "Matt Reeves"),
            ("Avatar: The Way of Water", "Sci-Fi", 2022, 7.6, "James Cameron"),
            ("Spider-Man: No Way Home", "Action", 2021, 8.2, "Jon Watts"),
            ("Avengers: Infinity War", "Action", 2018, 8.4, "Anthony Russo"),
            ("Rogue One: A Star Wars Story", "Sci-Fi", 2016, 7.8, "Gareth Edwards"),
            ("The Truman Show", "Drama", 1998, 8.2, "Peter Weir"),
            ("American Psycho", "Crime", 2000, 7.6, "Mary Harron"),
            ("Memento", "Mystery", 2000, 8.4, "Christopher Nolan"),
            ("Donnie Darko", "Sci-Fi", 2001, 8.0, "Richard Kelly"),
            ("Pan's Labyrinth", "Fantasy", 2006, 8.2, "Guillermo del Toro"),
            ("Amélie", "Romance", 2001, 8.3, "Jean-Pierre Jeunet"),
            ("2001: A Space Odyssey", "Sci-Fi", 1968, 8.3, "Stanley Kubrick"),
            ("Apocalypse Now", "War", 1979, 8.4, "Francis Ford Coppola"),
            ("Casablanca", "Romance", 1942, 8.5, "Michael Curtiz"),
            ("Psycho", "Horror", 1960, 8.5, "Alfred Hitchcock"),
            ("Rear Window", "Mystery", 1954, 8.5, "Alfred Hitchcock"),
            ("12 Angry Men", "Drama", 1957, 9.0, "Sidney Lumet"),
            ("The Pianist", "Drama", 2002, 8.5, "Roman Polanski"),
            ("Trainspotting", "Drama", 1996, 8.1, "Danny Boyle"),
            ("Catch Me If You Can", "Crime", 2002, 8.1, "Steven Spielberg"),
        ]

        # Load official TMDB theatrical posters dictionary
        official_posters_map = {}
        try:
            import json
            from pathlib import Path
            poster_file = Path(__file__).resolve().parent / "official_posters.json"
            if poster_file.exists():
                with open(poster_file, "r", encoding="utf-8") as pf:
                    official_posters_map = json.load(pf)
        except Exception:
            pass

        items = []
        for i in range(num_items):
            item_id = f"mov_{i+1:04d}"
            if i < len(famous_movies):
                title, genre, year, rating, director = famous_movies[i]
            else:
                extra_idx = i - len(famous_movies)
                genre = random.choice(["Sci-Fi", "Action", "Thriller", "Drama", "Crime", "Romance", "Animation"])
                title = f"{genre} Masterpiece #{extra_idx + 1}"
                year = random.randint(2000, 2026)
                rating = round(random.uniform(7.5, 9.2), 1)
                director = "Acclaimed Director"

            # Guaranteed official theatrical poster if in lookup, otherwise genre-matched
            poster_url = official_posters_map.get(title) or self._get_image_for_genre(genre, i)

            # 4 distinct artwork variants for Netflix Bandit Personalization:
            artwork_variants = {
                "variant_action": self.action_images[(i * 3 + 1) % len(self.action_images)],
                "variant_character": self.crime_thriller_images[(i * 2 + 4) % len(self.crime_thriller_images)],
                "variant_romantic": self.romance_drama_images[(i + 2) % len(self.romance_drama_images)],
                "variant_cinematic": poster_url,
            }

            items.append({
                "item_id": item_id,
                "title": title,
                "primary_genre": genre,
                "secondary_genre": "Drama" if genre != "Drama" else "Thriller",
                "release_year": year,
                "rating": rating,
                "director": director,
                "duration_min": random.randint(95, 175),
                "thumbnail_url": poster_url,
                "artwork_variants": artwork_variants,
                "tags": f"{genre},{director.replace(' ', '_')},{year}",
                "description": f"Critically acclaimed {genre} film directed by {director} ({year}). Rated {rating}/10."
            })

        return pd.DataFrame(items)

    # -------------------------------------------------------------
    # 2. MUSIC CATALOG (Spotify Style)
    # -------------------------------------------------------------
    def generate_music_catalog(self, num_items: int = 120) -> pd.DataFrame:
        famous_tracks = [
            ("Blinding Lights", "The Weeknd", "Pop"),
            ("Starboy", "The Weeknd ft. Daft Punk", "Electronic"),
            ("Get Lucky", "Daft Punk ft. Pharrell", "Electronic"),
            ("HUMBLE.", "Kendrick Lamar", "Hip-Hop"),
            ("Money Trees", "Kendrick Lamar", "Hip-Hop"),
            ("Paranoid Android", "Radiohead", "Rock"),
            ("Creep", "Radiohead", "Rock"),
            ("bad guy", "Billie Eilish", "Pop"),
            ("Time (Inception OST)", "Hans Zimmer", "Classical"),
            ("Cornfield Chase (Interstellar)", "Hans Zimmer", "Classical"),
            ("Do I Wanna Know?", "Arctic Monkeys", "Rock"),
            ("505", "Arctic Monkeys", "Rock"),
            ("So What", "Miles Davis", "Jazz"),
            ("Take Five", "Dave Brubeck", "Jazz"),
            ("Breathe Deeper", "Tame Impala", "Indie"),
            ("The Less I Know The Better", "Tame Impala", "Indie"),
            ("rumble", "Skrillex, Fred again..", "Electronic"),
            ("Danielle (smile on my face)", "Fred again..", "Electronic"),
            ("Levitating", "Dua Lipa", "Pop"),
            ("Don't Start Now", "Dua Lipa", "Pop"),
            ("Bohemian Rhapsody", "Queen", "Rock"),
            ("Hotel California", "Eagles", "Rock"),
            ("Smells Like Teen Spirit", "Nirvana", "Rock"),
            ("Lose Yourself", "Eminem", "Hip-Hop"),
            ("Midnight City", "M83", "Electronic"),
            ("Feel Good Inc.", "Gorillaz", "Alternative"),
            ("Clair de Lune", "Claude Debussy", "Classical"),
            ("Gymnopédie No. 1", "Erik Satie", "Ambient"),
            ("Weightless", "Marconi Union", "Ambient"),
            ("Resonance", "HOME", "Synthwave"),
        ]

        items = []
        for i in range(num_items):
            item_id = f"mus_{i+1:04d}"
            if i < len(famous_tracks):
                title, artist, genre = famous_tracks[i]
            else:
                genre = random.choice(["Electronic", "Indie", "Rock", "Hip-Hop", "Jazz", "Ambient"])
                title = f"{genre} Odyssey Track #{i - len(famous_tracks) + 1}"
                artist = f"Sound Architect {i+1}"

            thumbnail = self.music_images[i % len(self.music_images)]
            tempo = round(random.uniform(75, 145), 1)
            danceability = round(random.uniform(0.3, 0.95), 2)
            energy = round(random.uniform(0.3, 0.98), 2)

            items.append({
                "item_id": item_id,
                "title": title,
                "artist": artist,
                "genre": genre,
                "tempo": tempo,
                "danceability": danceability,
                "energy": energy,
                "valence": round(random.uniform(0.1, 0.9), 2),
                "duration_sec": random.randint(150, 310),
                "thumbnail_url": thumbnail,
                "tags": f"{genre},{artist.replace(' ', '_')}",
                "description": f"{genre} release by {artist}. Tempo: {tempo} BPM."
            })
        return pd.DataFrame(items)

    # -------------------------------------------------------------
    # 3. E-COMMERCE CATALOG (Amazon / Apple Style)
    # -------------------------------------------------------------
    def generate_ecommerce_catalog(self, num_items: int = 120) -> pd.DataFrame:
        famous_products = [
            ("Sony WH-1000XM5 Wireless Headphones", "Audio", "Sony", 399.99),
            ("Apple MacBook Pro 16\" M3 Max", "Computers", "Apple", 2499.99),
            ("Apple iPhone 16 Pro Max (Titanium)", "Electronics", "Apple", 1199.99),
            ("Sony PlayStation 5 Pro Console", "Gaming", "Sony", 699.99),
            ("Nintendo Switch OLED Model", "Gaming", "Nintendo", 349.99),
            ("Bose QuietComfort Ultra Earbuds", "Audio", "Bose", 299.99),
            ("Canon EOS R6 Mark II Mirrorless", "Electronics", "Canon", 2299.99),
            ("Dell UltraSharp 32\" 4K HDR Monitor", "Computers", "Dell", 899.99),
            ("Logitech MX Master 3S Wireless Mouse", "Computers", "Logitech", 99.99),
            ("Keychron Q1 Pro Custom Mechanical Keyboard", "Computers", "Keychron", 199.99),
            ("Sonos Era 300 Spatial Audio Speaker", "Audio", "Sonos", 449.99),
            ("Apple Watch Ultra 2 GPS + Cellular", "Wearables", "Apple", 799.99),
            ("Anker Prime 20,000mAh 200W Power Bank", "Electronics", "Anker", 129.99),
            ("Samsung 65\" OLED 4K Smart TV", "Electronics", "Samsung", 1799.99),
            ("Sennheiser HD 660S2 Audiophile Headphones", "Audio", "Sennheiser", 499.99),
        ]

        items = []
        for i in range(num_items):
            item_id = f"ecom_{i+1:04d}"
            if i < len(famous_products):
                title, category, brand, price = famous_products[i]
            else:
                category = random.choice(["Audio", "Computers", "Electronics", "Gaming", "Wearables"])
                brand = random.choice(["Sony", "Apple", "Logitech", "Bose", "Anker", "Samsung"])
                title = f"{brand} {category} Pro Edition #{i+1}"
                price = round(random.uniform(49.99, 1299.99), 2)

            thumbnail = self.ecom_images[i % len(self.ecom_images)]

            items.append({
                "item_id": item_id,
                "title": title,
                "category": category,
                "brand": brand,
                "price": price,
                "avg_rating": round(random.uniform(4.1, 4.9), 1),
                "num_reviews": random.randint(120, 8500),
                "thumbnail_url": thumbnail,
                "tags": f"{category},{brand},price_{int(price)}",
                "description": f"Flagship {category} item engineered by {brand}. Top rated by tech reviewers."
            })
        return pd.DataFrame(items)

    # -------------------------------------------------------------
    # 4. SHORT FEED CATALOG (TikTok Style)
    # -------------------------------------------------------------
    def generate_short_feed_catalog(self, num_items: int = 120) -> pd.DataFrame:
        viral_clips = [
            ("Tokyo Neon Cyberpunk Walk at 3AM", "Travel", "@tokyo_drift"),
            ("How Game Developers Fake 3D Lighting", "Tech", "@render_guru"),
            ("Secret 15-Minute Creamy Garlic Pasta", "Cooking", "@chef_marco"),
            ("Unreal Engine 5 Realism Tutorial", "Tech", "@cgi_master"),
            ("Crazy Calisthenics Street Progression", "Fitness", "@fit_pulse"),
            ("Cinematic Phone Color Grading Breakdown", "Tech", "@film_maker"),
            ("Synthwave 80s Beat in 60 Seconds", "Music", "@beat_lab"),
            ("Hidden iOS 18 Features You Never Knew", "Tech", "@daily_tech"),
            ("Satisfying Pottery Throwing ASMR", "ASMR", "@clay_craft"),
            ("5 Psychological Tricks to Learn Faster", "Education", "@mind_boost"),
        ]

        items = []
        for i in range(num_items):
            item_id = f"tok_{i+1:04d}"
            if i < len(viral_clips):
                title, topic, creator = viral_clips[i]
            else:
                topic = random.choice(["Tech", "Cooking", "Travel", "Gaming", "Fitness", "Design"])
                creator = f"@creator_{i+1}"
                title = f"Must-Watch {topic} Breakthrough #{i+1}"

            thumbnail = self.short_images[i % len(self.short_images)]

            items.append({
                "item_id": item_id,
                "title": title,
                "topic": topic,
                "creator": creator,
                "duration_sec": random.randint(15, 58),
                "sound_id": f"sound_{random.randint(1, 15):02d}",
                "view_count": random.randint(45000, 4800000),
                "thumbnail_url": thumbnail,
                "tags": f"{topic},{creator.replace('@', '')}",
                "description": f"Viral trending {topic} clip by {creator}."
            })
        return pd.DataFrame(items)

    # -------------------------------------------------------------
    # 5. USER PERSONAS & INTERACTION LOGS
    # -------------------------------------------------------------
    def generate_users(self, num_users: int = 80) -> pd.DataFrame:
        personas = [
            {"name": "SciFi_Tech_Geek", "fav_genres": ["Sci-Fi", "Thriller", "Action"], "fav_music": ["Electronic", "Rock"], "fav_ecom": ["Electronics", "Gaming"]},
            {"name": "Romance_Drama_Lover", "fav_genres": ["Romance", "Drama", "Comedy"], "fav_music": ["Pop", "Indie", "R&B"], "fav_ecom": ["Smart Home", "Audio"]},
            {"name": "Cinephile_Critic", "fav_genres": ["Drama", "Crime", "Mystery"], "fav_music": ["Jazz", "Classical"], "fav_ecom": ["Audio", "Computers"]},
            {"name": "Adrenaline_Junkie", "fav_genres": ["Action", "Horror", "War"], "fav_music": ["Rock", "Hip-Hop"], "fav_ecom": ["Gaming", "Wearables"]},
            {"name": "Family_Animation_Fan", "fav_genres": ["Animation", "Fantasy", "Sci-Fi"], "fav_music": ["Pop", "Ambient"], "fav_ecom": ["Smart Home", "Electronics"]}
        ]

        users = []
        for i in range(num_users):
            persona = random.choice(personas)
            users.append({
                "user_id": f"usr_{i+1:04d}",
                "username": f"{persona['name']}_{i+1}",
                "persona": persona["name"],
                "age": random.randint(19, 52),
                "fav_genres": ",".join(persona["fav_genres"]),
                "fav_music": ",".join(persona["fav_music"]),
                "fav_ecom": ",".join(persona["fav_ecom"]),
                "activity_level": random.choice(["high", "medium", "casual"])
            })
        return pd.DataFrame(users)

    def generate_interactions(
        self, users_df: pd.DataFrame, items_df: pd.DataFrame, domain: str = "movies"
    ) -> pd.DataFrame:
        interactions = []
        user_list = users_df.to_dict("records")
        item_list = items_df.to_dict("records")

        base_time = 1700000000

        for u in user_list:
            uid = u["user_id"]
            activity = u["activity_level"]
            num_actions = 45 if activity == "high" else (25 if activity == "medium" else 14)

            favs = u["fav_genres"].split(",") if domain == "movies" else (
                u["fav_music"].split(",") if domain == "music" else u["fav_ecom"].split(",")
            )

            for _ in range(num_actions):
                # 80% affinity to preferred genres
                if random.random() < 0.80:
                    filtered_items = [
                        it for it in item_list
                        if any(f in it.get("primary_genre", it.get("genre", it.get("category", ""))) for f in favs)
                    ]
                    if not filtered_items:
                        filtered_items = item_list
                    chosen_item = random.choice(filtered_items)
                    base_rating = random.uniform(3.8, 5.0)
                    watch_ratio = random.uniform(0.75, 1.0)
                    clicked = 1
                    liked = 1 if base_rating >= 4.2 else 0
                else:
                    chosen_item = random.choice(item_list)
                    base_rating = random.uniform(1.0, 3.8)
                    watch_ratio = random.uniform(0.1, 0.65)
                    clicked = 1 if random.random() < 0.6 else 0
                    liked = 0

                base_time += random.randint(45, 3600)

                interactions.append({
                    "user_id": uid,
                    "item_id": chosen_item["item_id"],
                    "rating": round(base_rating, 1),
                    "watch_ratio": round(watch_ratio, 2),
                    "clicked": clicked,
                    "liked": liked,
                    "timestamp": base_time,
                    "context_device": random.choice(["mobile", "tv", "desktop"]),
                    "context_hour": random.randint(0, 23)
                })

        df = pd.DataFrame(interactions)
        df = df.sort_values("timestamp").drop_duplicates(subset=["user_id", "item_id"], keep="last")
        return df.reset_index(drop=True)
