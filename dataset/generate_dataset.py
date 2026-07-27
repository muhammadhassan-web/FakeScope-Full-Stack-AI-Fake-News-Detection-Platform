"""
Dataset Generator for Fake News Detection
Generates a realistic labeled dataset with REAL and FAKE news samples.
In production, replace with: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
"""

import pandas as pd
import random
import csv
import os

random.seed(42)

REAL_NEWS = [
    ("Scientists discover new antibiotic that could fight drug-resistant bacteria", "Researchers at MIT have identified a novel antibiotic compound using machine learning, offering hope against MRSA and other resistant strains in peer-reviewed trials."),
    ("Federal Reserve raises interest rates by 0.25% amid inflation concerns", "The Federal Reserve increased benchmark rates for the third consecutive quarter, citing persistent consumer price inflation at 3.8% annually per official CPI data."),
    ("NASA's James Webb Telescope captures first images of distant exoplanet atmosphere", "The JWST detected carbon dioxide signatures in WASP-39b's atmosphere 700 light-years away, marking a breakthrough in exoplanet atmospheric characterization."),
    ("WHO reports global malaria cases decline for second consecutive year", "The World Health Organization published data showing a 6.8% reduction in malaria incidence across sub-Saharan Africa, attributed to expanded mosquito net distribution."),
    ("Electric vehicle sales surpass 10 million units globally in 2023", "International Energy Agency data confirms EV sales reached 10.5 million units last year, representing 14% of all new car purchases worldwide."),
    ("Supreme Court rules on landmark environmental protection case", "The court issued a 6-3 decision upholding EPA authority to regulate greenhouse gas emissions from power plants under the Clean Air Act."),
    ("Unemployment rate falls to 3.7% as economy adds 250,000 jobs", "The Bureau of Labor Statistics monthly report showed broad job gains across healthcare, technology, and manufacturing sectors last month."),
    ("New study links Mediterranean diet to reduced cardiovascular disease risk", "A 12-year longitudinal study of 45,000 participants published in NEJM found 28% lower heart disease incidence among strict Mediterranean diet adherents."),
    ("Government announces $2 billion infrastructure investment for rural broadband", "The bipartisan infrastructure law allocation will connect an estimated 4 million rural households to high-speed internet by 2026, officials confirmed."),
    ("International climate summit reaches agreement on carbon reduction targets", "Representatives from 192 nations signed a binding framework at COP31 committing to 45% emissions reductions by 2035 compared to 2005 baseline levels."),
    ("Breakthrough gene therapy shows promise for treating sickle cell disease", "Clinical trials at Johns Hopkins demonstrated 89% reduction in painful crises among patients receiving the CRISPR-based gene editing treatment."),
    ("Major bank reports quarterly earnings beat expectations by 12%", "The financial institution reported $4.2 billion net income for Q3, driven by strong investment banking fees and consumer lending growth."),
    ("City council approves new affordable housing development downtown", "The 400-unit mixed-income project received unanimous approval, with 30% units reserved for households earning below 60% of area median income."),
    ("Sports team wins championship after 15-year drought", "The city celebrated as the home team secured the title with a 4-2 series victory, ending the longest championship drought in franchise history."),
    ("Tech company announces 5,000 new jobs in regional headquarters expansion", "Following a $1.2 billion campus investment approved last spring, the firm confirmed hiring plans spanning software engineering, data science, and operations roles."),
    ("New public health guidelines recommend updated COVID booster schedule", "The CDC advisory committee voted to recommend annual autumn boosters for adults 65 and older, aligned with updated XBB.1.5-targeting formulations."),
    ("University researchers develop cheaper solar cell with record efficiency", "Perovskite-silicon tandem cells achieved 33.2% efficiency at a projected manufacturing cost of $0.18 per watt, beating previous benchmarks."),
    ("Local elections see record voter turnout in midterm cycle", "Election officials reported 58% participation in Tuesday's municipal contests, the highest rate in 24 years for an off-year election."),
    ("Pharmaceutical company receives FDA approval for new Alzheimer's drug", "The agency granted approval for lecanemab after trials showed 27% slower cognitive decline over 18 months in early-stage Alzheimer's patients."),
    ("Global food prices stabilize after two years of supply chain disruptions", "UN FAO food price index showed a 14-month consecutive decline, with wheat and vegetable oil prices returning to pre-pandemic levels."),
    ("Children's literacy rates improve following national reading initiative", "Department of Education data shows third-grade reading proficiency rose 4.2 percentage points nationwide after the 3-year phonics-based curriculum rollout."),
    ("Renewable energy now accounts for 30% of US electricity generation", "EIA monthly report confirms wind, solar, and hydro collectively exceeded natural gas generation for the first time in April on a monthly basis."),
    ("Central bank releases annual financial stability report", "The report highlighted improved bank capital ratios and declining non-performing loan rates but flagged commercial real estate as an emerging risk sector."),
    ("International trade agreement reduces tariffs on agricultural goods", "Trade partners finalized terms reducing duties on wheat, soybeans, and dairy by an average of 18%, effective upon parliamentary ratification."),
    ("Hospital system adopts AI tool to reduce diagnostic errors in radiology", "A 12-hospital network implemented deep learning software that reduced missed findings in chest CT scans by 31% in a prospective clinical validation study."),
]

FAKE_NEWS = [
    ("BREAKING: Government secretly putting mind-control chemicals in tap water, whistleblower reveals", "A former janitor at the CDC claims he discovered documents proving fluoride is actually a mind-control agent. The globalist elite have been dosing us for decades! Share before they delete this!"),
    ("EXPOSED: COVID vaccine contains microchips that activate with 5G signals, doctors confirm", "Brave doctors risking their careers have come forward with bombshell proof that mRNA vaccines contain nano-microchips that sync with 5G towers to track your location and thoughts in real time."),
    ("Alien spacecraft lands in Nevada desert, military cover-up exposed", "Multiple eyewitnesses report seeing a triangular craft land near Area 51. Our source inside the Pentagon says extraterrestrials have been living underground since 1952. Media blackout in effect!"),
    ("Secret cure for cancer suppressed by Big Pharma for 50 years finally leaked", "A natural compound found in apricot seeds was proven in 1972 to cure ALL cancers but pharmaceutical companies paid off the FDA to keep it hidden. This $3 remedy they don't want you to know about!"),
    ("Obama born in Kenya confirmed by new documents, sources say", "Leaked birth records reportedly obtained by investigators definitively prove what many have suspected. Multiple unnamed officials confirm the documents are authentic. The mainstream media won't cover this!"),
    ("George Soros funding antifa terrorism with $50 billion to overthrow America", "Financial records our investigative team obtained show the billionaire globalist transferring massive funds directly to antifa cell leaders in 47 cities. Democrats are complicit in this scheme!"),
    ("Military insiders say Biden was secretly replaced by a clone in 2020", "High-level Pentagon sources who cannot be named for their safety confirm the president is actually an AI-controlled body double. Behavioral analysts spot 12 discrepancies proving this shocking truth."),
    ("New world order documents reveal plan to reduce global population by 90%", "Leaked UN internal memos outline Agenda 2030's true purpose: eliminating 7 billion people through vaccines, chemtrails, and engineered famine. The elite will use climate change as cover."),
    ("Hollywood elites caught running child trafficking ring in pizza restaurant basement", "Our investigators have confirmed through anonymous sources that a major pizza chain is a front for a massive trafficking operation patronized by A-list celebrities and powerful politicians."),
    ("Doctors confirm drinking bleach cures coronavirus in 3 days, media silences truth", "A network of brave physicians claim that diluted bleach solution taken orally eliminates the COVID virus completely. The deep state is suppressing this $2 cure to force expensive vaccines on us."),
    ("Scientists admit climate change is completely fabricated hoax for carbon taxes", "An internal email chain from IPCC scientists caught red-handed admitting they falsified temperature data to justify the Green New Deal wealth transfer scheme. Follow the money!"),
    ("Deep state planning false flag attack to declare martial law before election", "Multiple whistleblowers inside DHS say operatives are planning a staged terror attack to cancel elections and declare emergency rule. They tried this in 2016 too — share this warning now!"),
    ("Mexican cartel members secretly voting in all 50 states, audit shows", "An election integrity group claims to have proof that millions of undocumented cartel members are registered to vote in swing states. The mainstream media is actively covering this up for Democrats."),
    ("Chemtrails confirmed: government admits spraying population with behavior-modifying toxins", "A FOIA request supposedly reveals that aircraft contrails contain lithium and other psychoactive substances designed to make populations docile and obedient to government control programs."),
    ("Elon Musk and Jeff Bezos part of secret lizard people society exposed by insider", "A former Bilderberg meeting attendee claims the world's wealthiest billionaires are actually reptilian shapeshifters who feed on human energy. Photographs analyzed by experts reveal telltale signs."),
    ("Voting machines in all swing states connected to Venezuela servers, flipped millions of votes", "Cybersecurity experts we cannot name have traced the election night anomalies directly to servers in Caracas. This mathematically impossible vote pattern is 100% proof of mass fraud!"),
    ("FDA secretly approves deadly experimental drug that causes spontaneous human combustion", "Suppressed FDA internal reports show a common blood pressure medication has triggered over 1,000 cases of spontaneous combustion but the pharmaceutical lobby keeps it on shelves for profits."),
    ("Pope Francis revealed to be undercover Satanist running occult rituals in Vatican", "A former Swiss Guard has gone into hiding after leaking proof of Satanic ceremonies in secret Vatican chambers attended by senior cardinals. The globalist church is their vehicle for world domination."),
    ("Bill Gates patents sun-blocking technology to cause global famine and mass death", "Microsoft's founder has obtained patents for stratospheric aerosol injection designed to block sunlight and crash global food production as part of his population reduction agenda. Boycott his products!"),
    ("NASA caught photoshopping moon landing footage, insider spills the beans", "A retired photo technician from the 1960s claims the entire Apollo program was staged in a New Mexico warehouse by Stanley Kubrick. Newly enhanced photographs show lighting inconsistencies proving the hoax."),
    ("5G towers confirmed to spread COVID-19 through electromagnetic frequency modulation", "An engineer who worked on 5G infrastructure reveals the towers are programmed to emit specific frequencies that activate the COVID virus in people with nano-particles from vaccines. Connect the dots!"),
    ("Democrats plan to confiscate all guns by end of next month under secret executive order", "A source inside the White House reveals Biden has signed a secret executive order that will activate FEMA gun confiscation teams nationwide. Stockpile ammunition and share this before it's taken down!"),
    ("Mega-earthquake to destroy California on Tuesday confirmed by USGS geologists", "Geologists who are afraid to speak publicly confirm a 9.8 magnitude quake will devastate the entire West Coast this Tuesday. The government is hiding this to prevent panic. Evacuate NOW!"),
    ("Fluoride in water lowering IQ to make population easier to control, Harvard study confirms", "A Harvard meta-analysis was buried because it proved fluoridated water lowers children's IQ by 7 points on average, part of a long-running depopulation and population management eugenics program."),
    ("George Floyd alive in witness protection program, death was staged psyop", "An investigative reporter has obtained records from the Marshals Service proving the incident was staged to trigger race riots and justify defunding police as part of the socialist takeover agenda."),
]

def generate_dataset(n_real=500, n_fake=500, output_path="news_dataset.csv"):
    rows = []
    
    for i in range(n_real):
        base = REAL_NEWS[i % len(REAL_NEWS)]
        title = base[0]
        text = base[1]
        # Add slight variations
        if i >= len(REAL_NEWS):
            title = title + f" — Updated Report {i}"
        rows.append({"id": i, "title": title, "text": text, "label": "REAL"})
    
    for i in range(n_fake):
        base = FAKE_NEWS[i % len(FAKE_NEWS)]
        title = base[0]
        text = base[1]
        if i >= len(FAKE_NEWS):
            title = title + f" (Part {i})"
        rows.append({"id": n_real + i, "title": title, "text": text, "label": "FAKE"})
    
    random.shuffle(rows)
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Dataset saved: {output_path} | REAL: {n_real} | FAKE: {n_fake} | Total: {n_real+n_fake}")
    return df

if __name__ == "__main__":
    os.makedirs("dataset", exist_ok=True)
    generate_dataset(500, 500, "dataset/news_dataset.csv")
