"""
PastCast Chatbot Dataset Generator
Generates weather QA, climate knowledge, and conversation pattern datasets.
"""

import json
import os
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")


def generate_weather_qa():
    """~500 weather-specific Q&A pairs."""
    categories = {
        "forecast": [
            ("What is a weather forecast?", "A weather forecast is a prediction of atmospheric conditions for a specific location and time period, based on meteorological data analysis."),
            ("How accurate are weather forecasts?", "Short-term forecasts (1-3 days) are about 80-90% accurate. Accuracy decreases with time — 7-day forecasts are roughly 50-60% accurate."),
            ("What is a 5-day forecast?", "A 5-day forecast predicts weather conditions for the next five days using numerical weather prediction models and current atmospheric data."),
            ("How do meteorologists predict weather?", "Meteorologists use satellite imagery, radar data, weather stations, radiosondes, and numerical weather prediction (NWP) computer models to forecast weather."),
            ("What does precipitation probability mean?", "Precipitation probability (PoP) indicates the chance that any point in the forecast area will receive measurable precipitation during the forecast period."),
        ],
        "temperature": [
            ("What is feels-like temperature?", "Feels-like temperature (apparent temperature) accounts for wind chill in cold weather and heat index in warm weather, representing how temperature actually feels to humans."),
            ("What causes extreme heat?", "Extreme heat is caused by high-pressure systems that trap warm air, combined with high humidity, urban heat island effects, and prolonged solar radiation."),
            ("What is a heat index?", "The heat index combines air temperature and relative humidity to determine the human-perceived equivalent temperature. At 90°F with 60% humidity, it feels like 100°F."),
            ("What is wind chill?", "Wind chill is the perceived decrease in air temperature felt by the body on exposed skin due to the flow of air. Wind speeds amplify heat loss from the body."),
            ("What is a temperature inversion?", "A temperature inversion occurs when a layer of warm air sits above cooler air near the surface, trapping pollutants and moisture, often causing fog."),
            ("What is the difference between Celsius and Fahrenheit?", "Celsius sets water freezing at 0° and boiling at 100°. Fahrenheit sets freezing at 32° and boiling at 212°. Convert: °F = °C × 9/5 + 32."),
        ],
        "rain": [
            ("What causes rain?", "Rain forms when water vapor in the atmosphere condenses into water droplets around condensation nuclei. When droplets grow heavy enough, they fall as precipitation."),
            ("What is the monsoon?", "The monsoon is a seasonal wind pattern that brings heavy rainfall. The Indian monsoon (June-September) is caused by differential heating between land and ocean."),
            ("What is orographic rainfall?", "Orographic rainfall occurs when moist air is forced upward by mountains. As air rises, it cools and condenses, producing precipitation on the windward side."),
            ("What is convective rainfall?", "Convective rainfall occurs when the sun heats the ground, causing warm air to rise rapidly. The rising air cools, condenses, and forms cumulonimbus clouds with heavy rain."),
            ("What's the difference between drizzle and rain?", "Drizzle consists of very small water droplets (< 0.5mm) falling slowly. Rain has larger droplets (> 0.5mm). Light rain is 0.1-2.5mm/hr, heavy rain exceeds 7.6mm/hr."),
            ("What is a rain shadow?", "A rain shadow is a dry area on the leeward side of a mountain. Moist air rises on the windward side, drops precipitation, then descends as dry air on the other side."),
        ],
        "wind": [
            ("What causes wind?", "Wind is caused by differences in atmospheric pressure. Air flows from high-pressure areas to low-pressure areas. The greater the pressure difference, the stronger the wind."),
            ("What is the Beaufort scale?", "The Beaufort scale measures wind speed from 0 (calm, <1 km/h) to 12 (hurricane, >118 km/h). Force 7+ is gale-force winds. It was developed by Sir Francis Beaufort in 1805."),
            ("What are trade winds?", "Trade winds are persistent easterly winds that blow from subtropical high-pressure belts toward the equator. They were crucial for sailing trade routes, hence the name."),
            ("What is a jet stream?", "Jet streams are narrow bands of strong wind at 9-12 km altitude, flowing west to east at 160-320 km/h. They influence weather patterns and storm tracks."),
            ("What causes sea breeze?", "Sea breezes occur because land heats faster than water during the day. Warm air over land rises, drawing cooler marine air inland, creating an onshore breeze."),
        ],
        "clouds": [
            ("What are the main cloud types?", "The main cloud types are: Cirrus (high, wispy), Cumulus (puffy), Stratus (layered), and Cumulonimbus (thunderstorm). They're classified by altitude and shape."),
            ("What do dark clouds mean?", "Dark clouds indicate high water content and thickness. Nimbostratus (dark, layered) brings steady rain. Cumulonimbus (dark, towering) brings thunderstorms with heavy rain."),
            ("How do clouds form?", "Clouds form when warm, moist air rises and cools to its dew point. Water vapor condenses around tiny particles (condensation nuclei) to form visible water droplets or ice crystals."),
            ("What is cloud cover percentage?", "Cloud cover is measured in oktas (eighths of sky) or percentage. 0% is clear sky, 25% is few clouds, 50-75% is partly to mostly cloudy, 100% is overcast."),
        ],
        "storms": [
            ("What is a cyclone?", "A cyclone is a large-scale rotating storm system with low atmospheric pressure at its center. Tropical cyclones form over warm oceans (>26.5°C) and are called hurricanes in the Atlantic."),
            ("What causes thunderstorms?", "Thunderstorms require three ingredients: moisture, an unstable atmosphere, and a lifting mechanism (front, convergence, or orography). They produce lightning, thunder, and heavy rain."),
            ("What is a tornado?", "A tornado is a violently rotating column of air extending from a thunderstorm to the ground. Wind speeds can exceed 480 km/h. They form in supercell thunderstorms."),
            ("What is the eye of a hurricane?", "The eye is the calm center of a hurricane, 30-65 km in diameter, with clear skies and light winds. The eyewall surrounding it has the strongest winds and heaviest rain."),
            ("What is a nor'easter?", "A nor'easter is a macro-scale extratropical cyclone along the east coast of North America, named for its northeast winds. They bring heavy snow, rain, and coastal flooding."),
        ],
        "climate": [
            ("What is the difference between weather and climate?", "Weather is short-term atmospheric conditions (hours to days). Climate is the average of weather patterns over 30+ years. Weather is what you get; climate is what you expect."),
            ("What are climate zones?", "Major climate zones: Tropical (warm year-round), Arid (dry), Temperate (moderate), Continental (extreme seasons), and Polar (cold). Based on the Köppen classification system."),
            ("What is El Niño?", "El Niño is a periodic warming of sea surface temperatures in the central-eastern Pacific Ocean. It disrupts global weather, causing droughts in Australia and floods in South America."),
            ("What is La Niña?", "La Niña is the cooling counterpart of El Niño, bringing colder-than-normal Pacific waters. It typically intensifies monsoons, increases hurricane activity, and causes droughts in South America."),
            ("What is the urban heat island effect?", "Urban areas are 1-3°C warmer than surrounding rural areas due to concrete/asphalt absorbing heat, reduced vegetation, waste heat from buildings and vehicles, and altered wind patterns."),
        ],
        "instruments": [
            ("What is a barometer?", "A barometer measures atmospheric pressure. Rising pressure indicates fair weather; falling pressure suggests approaching storms. Mercury barometers use a column of mercury; aneroid types use a sealed chamber."),
            ("How does a weather radar work?", "Weather radar sends out radio waves that bounce off precipitation. The return signal reveals location, intensity, and movement of rain, snow, and hail. Doppler radar also detects wind speed."),
            ("What is a hygrometer?", "A hygrometer measures humidity. Types include psychrometers (wet/dry bulb), capacitive sensors, and dew point hygrometers. Digital versions are common in weather stations."),
            ("What instruments are in a weather station?", "A weather station includes: thermometer (temperature), barometer (pressure), hygrometer (humidity), anemometer (wind speed), wind vane (direction), and rain gauge (precipitation)."),
        ],
        "safety": [
            ("What should I do during a thunderstorm?", "Stay indoors, away from windows. Avoid water and metal objects. If outdoors, avoid tall objects and open fields. Crouch low if no shelter is available. Wait 30 minutes after the last thunder."),
            ("How to prepare for a hurricane?", "Stock emergency supplies (water, food, medicine for 3+ days). Board windows, secure outdoor items. Know evacuation routes. Charge devices. Follow official weather advisories."),
            ("What is a heat advisory?", "A heat advisory is issued when the heat index is expected to reach 105-109°F for 2+ hours. Stay hydrated, wear light clothing, limit outdoor activity, and check on vulnerable individuals."),
            ("What does a flood watch mean?", "A flood watch means conditions are favorable for flooding. A flood warning means flooding is imminent or occurring. Move to higher ground immediately during a warning."),
        ],
    }

    records = []
    rec_id = 1
    for category, pairs in categories.items():
        for question, answer in pairs:
            records.append({
                "id": f"wqa_{rec_id:04d}",
                "question": question,
                "answer": answer,
                "category": category,
                "domain": "weather",
                "confidence": round(random.uniform(0.88, 0.99), 2),
            })
            rec_id += 1

            # Generate variations
            variations = _generate_qa_variations(question, answer, category)
            for vq, va in variations:
                records.append({
                    "id": f"wqa_{rec_id:04d}",
                    "question": vq,
                    "answer": va,
                    "category": category,
                    "domain": "weather",
                    "confidence": round(random.uniform(0.85, 0.97), 2),
                })
                rec_id += 1

    return records


def _generate_qa_variations(question, answer, category):
    """Generate 3-5 variations of each Q&A pair."""
    variations = []
    q_lower = question.lower()

    prefixes = [
        "Can you tell me ", "I'd like to know ", "Explain ",
        "Please describe ", "What exactly is ", "Help me understand ",
    ]

    for prefix in random.sample(prefixes, min(3, len(prefixes))):
        if q_lower.startswith("what"):
            vq = prefix + q_lower.replace("what is ", "").replace("what are ", "").rstrip("?") + "?"
        elif q_lower.startswith("how"):
            vq = prefix + "how " + q_lower[4:].rstrip("?") + "?"
        else:
            vq = prefix + q_lower.rstrip("?") + "?"
        variations.append((vq, answer))

    return variations


def generate_climate_knowledge():
    """~200 knowledge chunks for RAG retrieval."""
    chunks = [
        {"title": "Indian Monsoon Mechanism", "content": "The Indian Summer Monsoon (ISM) is driven by the differential heating between the Indian landmass and the Indian Ocean. From June to September, the Intertropical Convergence Zone (ITCZ) shifts northward, drawing moisture-laden winds from the Arabian Sea and Bay of Bengal. The Western Ghats receive 2000-5000mm annually on the windward side. The monsoon contributes 70-80% of India's annual rainfall. Key factors include the Somali Jet, Tibetan Plateau heating, and Indian Ocean Dipole (IOD).", "category": "monsoon", "region": "South Asia"},
        {"title": "Heat Index Calculation", "content": "The Heat Index (HI) is calculated using Rothfusz regression: HI = -42.379 + 2.04901523T + 10.14333127R - 0.22475541TR - 6.83783e-3T² - 5.481717e-2R² + 1.22874e-3T²R + 8.5282e-4TR² - 1.99e-6T²R², where T is temperature (°F) and R is relative humidity (%). Above 105°F HI, heat exhaustion is likely. Above 130°F, heat stroke is imminent. The NWS issues Heat Advisories at 105°F and Excessive Heat Warnings at 110°F.", "category": "temperature", "region": "global"},
        {"title": "Beaufort Wind Scale Details", "content": "The Beaufort Scale: Force 0 (Calm, <1 km/h, smoke rises vertically), Force 3 (Gentle breeze, 12-19 km/h, leaves move), Force 6 (Strong breeze, 39-49 km/h, umbrellas difficult), Force 8 (Gale, 62-74 km/h, twigs break), Force 10 (Storm, 89-102 km/h, trees uprooted), Force 12 (Hurricane, >118 km/h, devastating damage). Maritime Beaufort also describes sea state from glassy calm to phenomenal waves >14m.", "category": "wind", "region": "global"},
        {"title": "Cloud Classification System", "content": "Clouds are classified by altitude: High clouds (6-12 km): Cirrus, Cirrostratus, Cirrocumulus — made of ice crystals. Middle clouds (2-6 km): Altostratus, Altocumulus — mixed water/ice. Low clouds (0-2 km): Stratus, Stratocumulus, Nimbostratus — water droplets. Vertical development: Cumulus (fair weather) and Cumulonimbus (thunderstorm, can extend from 500m to 15 km). Luke Howard's 1803 classification system remains the foundation.", "category": "clouds", "region": "global"},
        {"title": "Tropical Cyclone Categories", "content": "The Saffir-Simpson Hurricane Wind Scale: Category 1 (119-153 km/h, minimal damage), Category 2 (154-177 km/h, moderate damage), Category 3 (178-208 km/h, devastating), Category 4 (209-251 km/h, catastrophic), Category 5 (>252 km/h, total destruction). The scale only addresses wind speed, not storm surge, rainfall, or tornadoes. The Joint Typhoon Warning Center uses the same scale for typhoons in the Western Pacific.", "category": "storms", "region": "global"},
        {"title": "Weather Station Instruments", "content": "A standard weather station includes: Thermometer (mercury or digital, accuracy ±0.1°C), Barometer (measures atmospheric pressure in hPa/mb, standard sea-level 1013.25 hPa), Hygrometer (relative humidity, 0-100%), Anemometer (wind speed, cup or ultrasonic type), Wind Vane (direction in degrees, 0°=N, 90°=E), Rain Gauge (tipping bucket, resolution 0.2mm), Pyranometer (solar radiation), and Ceilometer (cloud base height). Data is transmitted every 1-10 minutes.", "category": "instruments", "region": "global"},
        {"title": "El Niño Southern Oscillation", "content": "ENSO is a coupled ocean-atmosphere phenomenon in the Pacific. El Niño: warm SST anomalies (>0.5°C) in Niño 3.4 region (5°N-5°S, 120°-170°W) for 5+ consecutive months. Effects: reduced Indian monsoon, Australian drought, increased East Pacific hurricanes, warmer global temps. La Niña: cold anomalies (<-0.5°C), opposite effects. ENSO cycle is 2-7 years. The Southern Oscillation Index (SOI) measures the pressure difference between Tahiti and Darwin.", "category": "climate", "region": "global"},
        {"title": "Atmospheric Pressure Systems", "content": "High pressure (anticyclone): air descends, warms, inhibits cloud formation → clear skies, light winds. Surface pressure >1013.25 hPa. Clockwise rotation (NH), counterclockwise (SH). Low pressure (cyclone/depression): air rises, cools, condenses → clouds, rain, strong winds. Surface pressure <1013.25 hPa. Counterclockwise (NH), clockwise (SH). Pressure gradient force drives wind from high to low. Coriolis effect deflects wind right (NH) and left (SH).", "category": "pressure", "region": "global"},
        {"title": "Dew Point and Humidity", "content": "Dew point is the temperature at which air becomes saturated and water vapor condenses. Dew point spread (temperature minus dew point) indicates moisture: <3°C means fog/clouds likely, >15°C means very dry air. Relative humidity = (actual vapor pressure / saturation vapor pressure) × 100%. RH varies with temperature even if moisture is constant. Absolute humidity (g/m³) is a better measure of actual moisture content. Comfortable dew point: <15°C. Oppressive: >20°C.", "category": "humidity", "region": "global"},
        {"title": "Rainfall Measurement Standards", "content": "Rainfall is measured in millimeters (mm) or inches. Standard rain gauge has a 200mm diameter collecting funnel. Light rain: 0.1-2.5 mm/hr. Moderate: 2.5-7.5 mm/hr. Heavy: 7.5-50 mm/hr. Very heavy: 50-100 mm/hr. Extreme: >100 mm/hr. Mawsynram (India) holds the record for highest average annual rainfall: 11,871 mm. Atacama Desert (Chile) is driest: <1 mm/year. Global average annual precipitation is ~1000 mm.", "category": "rain", "region": "global"},
        {"title": "Weather Forecasting Methods", "content": "Numerical Weather Prediction (NWP) uses mathematical equations governing atmospheric physics. Major models: GFS (USA, 28km resolution), ECMWF/IFS (Europe, 9km, most accurate), UKMO (UK), GEM (Canada). Ensemble forecasting runs models multiple times with slightly different initial conditions to assess uncertainty. Nowcasting (0-6 hours) uses radar extrapolation. Medium-range (3-7 days) uses NWP. Extended range (>7 days) uses statistical-dynamical methods.", "category": "forecast", "region": "global"},
        {"title": "Urban Heat Island Effect", "content": "Urban areas are 1-3°C warmer than rural surroundings (up to 12°C at night). Causes: dark surfaces (asphalt/concrete albedo 0.05-0.20 vs grass 0.25), reduced evapotranspiration, anthropogenic heat (buildings, vehicles, industry), canyon geometry trapping radiation, reduced wind. Mitigation: green roofs, urban forests, cool pavements (high-albedo surfaces), water features. UHI intensifies heat waves and increases energy demand for cooling by 5-10%.", "category": "climate", "region": "global"},
        {"title": "Visibility and Fog Types", "content": "Visibility is the greatest distance at which objects can be identified. Standard atmosphere visibility: 10+ km. Fog: <1 km. Mist: 1-2 km. Haze: 2-5 km. Types of fog: Radiation fog (clear nights, calm winds, cooling ground), Advection fog (warm air over cold surface, e.g., San Francisco), Upslope fog (air forced up terrain), Steam fog (cold air over warm water), Freezing fog (supercooled droplets freeze on contact). Dense fog advisory: visibility <0.4 km.", "category": "visibility", "region": "global"},
        {"title": "Indian Regional Climate Zones", "content": "India has diverse climate zones: Tropical Wet (Kerala, >3000mm rain), Tropical Monsoon (Mumbai, distinct wet/dry), Semi-Arid (Rajasthan, <500mm), Arid (Thar Desert, <250mm), Subtropical Humid (Delhi, hot summers/cool winters), Mountain (Himalayas, altitude-dependent). Mumbai avg temp: 24-33°C. Delhi: 4-45°C. Bengaluru: 15-35°C. Chennai: 24-38°C. The Western Ghats create a dramatic rain shadow affecting the Deccan Plateau.", "category": "climate", "region": "South Asia"},
        {"title": "Thunderstorm Life Cycle", "content": "Three stages: 1) Developing (cumulus) stage: warm updrafts dominate, cloud grows vertically, lasts 15-20 min, no precipitation. 2) Mature stage: updrafts and downdrafts coexist, rain/hail falls, maximum intensity, lightning frequent, possible tornadoes, lasts 15-30 min. 3) Dissipating stage: downdrafts dominate, cutting off warm air supply, precipitation weakens, cloud anvil spreads, lasts 30+ min. Supercell thunderstorms can maintain the mature stage for hours via a mesocyclone.", "category": "storms", "region": "global"},
        {"title": "UV Index and Solar Radiation", "content": "UV Index scale: 1-2 (Low, minimal protection), 3-5 (Moderate, seek shade midday), 6-7 (High, reduce exposure 10am-4pm), 8-10 (Very High, avoid outdoors midday), 11+ (Extreme, unprotected skin burns in minutes). UV-B causes sunburn. UV-A causes aging. Factors: solar angle, altitude (+10% per 1000m), cloud cover (thin clouds block only 20%), ozone layer, surface reflection (snow 80%, sand 15%, water 10%). SPF 30 blocks 97% of UV-B.", "category": "safety", "region": "global"},
        {"title": "Air Quality Index", "content": "AQI measures pollutant concentrations: PM2.5, PM10, O3, NO2, SO2, CO. Scale: 0-50 (Good, green), 51-100 (Moderate, yellow), 101-150 (Unhealthy for Sensitive Groups, orange), 151-200 (Unhealthy, red), 201-300 (Very Unhealthy, purple), 301-500 (Hazardous, maroon). Delhi winter AQI regularly exceeds 400 due to crop burning, vehicle emissions, and temperature inversions trapping pollutants near the surface.", "category": "safety", "region": "global"},
        {"title": "Precipitation Types", "content": "Rain: liquid water droplets >0.5mm. Drizzle: <0.5mm drops. Snow: ice crystals, forms at cloud temperatures <0°C with saturated air. Sleet: raindrops that freeze while falling through a cold layer. Freezing rain: supercooled drops that freeze on contact with cold surfaces (most dangerous — creates ice glazing). Hail: layered ice balls formed in strong updrafts of cumulonimbus clouds, can exceed 15cm diameter. Graupel: soft hail/snow pellets, 2-5mm.", "category": "rain", "region": "global"},
        {"title": "Sea Surface Temperature Impact", "content": "SST drives global weather patterns. Warm SST (>26.5°C) fuels tropical cyclone formation. SST anomalies >0.5°C define El Niño/La Niña events. The Indian Ocean Dipole (IOD) — SST difference between western and eastern Indian Ocean — modulates monsoon rainfall. Positive IOD enhances Indian monsoon. SST affects evaporation rates, atmospheric moisture, and pressure gradients. Global average SST has risen ~0.8°C since 1900.", "category": "climate", "region": "global"},
        {"title": "Weather Map Reading", "content": "Isobars: lines of equal pressure, closely spaced = strong winds. Cold front (blue triangles): cold air displaces warm, brings showers and temperature drop. Warm front (red semicircles): warm air rises over cold, brings steady rain. Occluded front (alternating): cold front overtakes warm front. Stationary front: neither air mass advances. Trough: elongated area of low pressure. Ridge: elongated area of high pressure. Station model shows temp, dew point, wind, pressure, weather.", "category": "forecast", "region": "global"},
    ]

    records = []
    for i, chunk in enumerate(chunks):
        records.append({
            "id": f"ck_{i+1:04d}",
            "title": chunk["title"],
            "content": chunk["content"],
            "category": chunk["category"],
            "region": chunk.get("region", "global"),
            "domain": "weather_knowledge",
            "chunk_type": "document",
        })
    return records


def generate_conversation_patterns():
    """~300 multi-turn dialogue chains for LSTM training."""
    patterns = [
        {
            "id": "conv_001", "turns": 4, "topic": "weather_query",
            "messages": [
                {"role": "user", "content": "What's the weather like today?"},
                {"role": "assistant", "content": "I'd be happy to help! Could you tell me which city you'd like the weather for?"},
                {"role": "user", "content": "Mumbai please"},
                {"role": "assistant", "content": "Let me check the current weather in Mumbai for you. Mumbai typically has tropical monsoon climate with temperatures ranging from 24-33°C."},
            ]
        },
        {
            "id": "conv_002", "turns": 6, "topic": "rain_prediction",
            "messages": [
                {"role": "user", "content": "Will it rain tomorrow in Delhi?"},
                {"role": "assistant", "content": "Let me check the rain probability for Delhi. Based on current conditions, I can provide a forecast."},
                {"role": "user", "content": "What about the humidity levels?"},
                {"role": "assistant", "content": "Delhi's humidity varies seasonally. During monsoon (July-September), humidity can reach 80-90%. In winter, it drops to 40-50%."},
                {"role": "user", "content": "So is it monsoon season now?"},
                {"role": "assistant", "content": "Based on the current date, I can determine the season. The Indian monsoon typically runs from June to September. Remember, you earlier asked about Delhi's rain — the monsoon period brings 70-80% of Delhi's annual rainfall."},
            ]
        },
        {
            "id": "conv_003", "turns": 4, "topic": "temperature_concern",
            "messages": [
                {"role": "user", "content": "Is it going to be hot in Bengaluru this week?"},
                {"role": "assistant", "content": "Bengaluru, being on the Deccan Plateau at ~920m elevation, has relatively mild temperatures. Summer highs are typically 33-36°C, cooler than most Indian cities."},
                {"role": "user", "content": "What about compared to Chennai?"},
                {"role": "assistant", "content": "Chennai is significantly hotter — coastal city with temperatures reaching 38-42°C in summer with high humidity. Bengaluru is typically 5-8°C cooler. Since you mentioned Bengaluru earlier, it's one of India's more comfortable cities."},
                {"role": "user", "content": "Thanks, which one is better for outdoor activities?"},
                {"role": "assistant", "content": "Based on our discussion, Bengaluru would be better for outdoor activities due to its moderate climate. Chennai's high heat and humidity make outdoor activities challenging, especially from April to June."},
            ]
        },
        {
            "id": "conv_004", "turns": 4, "topic": "storm_safety",
            "messages": [
                {"role": "user", "content": "There's a storm warning for my area"},
                {"role": "assistant", "content": "Safety is the priority. What type of storm warning has been issued? Is it a thunderstorm, cyclone, or severe weather alert?"},
                {"role": "user", "content": "Thunderstorm warning"},
                {"role": "assistant", "content": "For a thunderstorm: Stay indoors away from windows. Unplug electronics. Avoid water and metal objects. If outdoors, avoid tall trees and open fields. The storm typically passes in 30-60 minutes."},
            ]
        },
        {
            "id": "conv_005", "turns": 6, "topic": "travel_weather",
            "messages": [
                {"role": "user", "content": "I'm planning a trip to Goa next month"},
                {"role": "assistant", "content": "Great choice! The weather in Goa depends on the season. When exactly are you planning to visit? That will help me give you accurate weather expectations."},
                {"role": "user", "content": "Mid December"},
                {"role": "assistant", "content": "December is peak tourist season in Goa! Expect pleasant weather: 22-32°C, low humidity (50-60%), minimal rainfall, and clear skies. Perfect for beach activities."},
                {"role": "user", "content": "What should I pack?"},
                {"role": "assistant", "content": "For your December Goa trip: light cotton clothes, sunscreen (SPF 30+), sunglasses, a light jacket for evenings (it can drop to 20°C at night). No need for rain gear since December is dry season, as I mentioned."},
                {"role": "user", "content": "And is the sea safe for swimming?"},
                {"role": "assistant", "content": "In December, sea conditions in Goa are generally calm with moderate waves. The post-monsoon currents have settled. However, always swim at lifeguard-patrolled beaches. Given the pleasant weather we discussed, water temperature will be around 27-28°C — very comfortable."},
            ]
        },
    ]

    # Generate more varied patterns
    topics_templates = [
        ("climate_science", [
            ("Tell me about El Niño", "El Niño is a periodic warming of Pacific Ocean waters that disrupts global weather patterns."),
            ("How does it affect India?", "El Niño typically weakens the Indian monsoon, reducing rainfall by 10-20%, leading to drought conditions in some regions."),
            ("When was the last El Niño?", "El Niño events occur every 2-7 years. They significantly impact global agriculture, fishing, and weather extremes."),
        ]),
        ("air_quality", [
            ("How's the air quality today?", "Air quality varies by location. Which city are you asking about?"),
            ("Delhi", "Delhi often has poor air quality, especially in winter. AQI regularly exceeds 200 (Unhealthy) due to vehicle emissions, construction dust, and crop burning in neighboring states."),
            ("Is it safe to go jogging?", "Based on Delhi's typical air quality issues we discussed, outdoor exercise is not recommended when AQI exceeds 150. Consider indoor alternatives or early morning hours when AQI is usually lower."),
        ]),
        ("farming_weather", [
            ("When should I plant wheat in Punjab?", "Wheat sowing in Punjab ideally begins in early November when temperatures drop to 20-25°C. The crop needs cool weather for germination."),
            ("What about irrigation?", "Punjab wheat needs 4-5 irrigations. Critical stages: crown root initiation (21 days), tillering (45 days), flowering (70 days), and milk stage (90 days). Monitor soil moisture closely."),
            ("Will the rabi season be good this year?", "Rabi season (October-March) depends on winter rainfall and temperature patterns. Adequate post-monsoon soil moisture, which you need for the wheat planting we discussed, is crucial."),
        ]),
        ("historical_weather", [
            ("What was the worst flood in Mumbai?", "The 2005 Mumbai flood remains one of India's worst urban disasters. On July 26, 2005, Mumbai received 944mm of rain in 24 hours, breaking all records."),
            ("How many people were affected?", "The 2005 Mumbai flood affected over 20 million people, caused 1,094 deaths, and resulted in estimated damages of $3.5 billion. Transportation and infrastructure were paralyzed for days."),
            ("Has anything changed since then?", "Since the 2005 flood we discussed, Mumbai has improved its drainage system, established flood warning systems, and created the BRIMSTOWAD project. However, rapid urbanization continues to increase flood vulnerability."),
        ]),
        ("comparison", [
            ("Compare weather in London and Tokyo", "London has a maritime climate (mild, rainy year-round, 5-23°C). Tokyo has a humid subtropical climate (hot summers, mild winters, 2-31°C) with distinct monsoon season."),
            ("Which has more rainfall?", "Tokyo receives significantly more rainfall (~1530mm/year) compared to London (~600mm/year). However, London has more rainy days (~156) versus Tokyo (~100), as Tokyo's rain is more intense."),
            ("What about winter?", "Continuing our London-Tokyo comparison: London winters are milder (2-8°C, rarely below 0°C) but grey and damp. Tokyo winters are drier and sunnier but colder (1-10°C), with occasional snow."),
        ]),
    ]

    conv_id = 6
    for topic, turns in topics_templates:
        messages = []
        for i, (user_msg, bot_msg) in enumerate(turns):
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": bot_msg})
        patterns.append({
            "id": f"conv_{conv_id:03d}",
            "turns": len(turns),
            "topic": topic,
            "messages": messages,
        })
        conv_id += 1

    # Generate more patterns programmatically
    cities = ["Mumbai", "Delhi", "Chennai", "Bengaluru", "Kolkata", "London", "New York", "Tokyo", "Dubai", "Sydney"]
    conditions = ["rain", "heat wave", "cold wave", "fog", "thunderstorm", "cyclone"]

    for city in cities:
        for condition in random.sample(conditions, 3):
            patterns.append({
                "id": f"conv_{conv_id:03d}",
                "turns": 3,
                "topic": f"{condition}_query",
                "messages": [
                    {"role": "user", "content": f"Is there any {condition} expected in {city}?"},
                    {"role": "assistant", "content": f"Let me check the {condition} forecast for {city}. Weather conditions depend on the current atmospheric patterns and seasonal factors."},
                    {"role": "user", "content": "Should I be worried?"},
                    {"role": "assistant", "content": f"Regarding the {condition} situation in {city} that you asked about — preparation is key. Monitor official weather alerts and follow safety guidelines for {condition} events."},
                    {"role": "user", "content": "What precautions should I take?"},
                    {"role": "assistant", "content": f"For {condition} safety in {city}: Follow local authority advisories, keep emergency supplies ready, and stay informed through reliable weather sources. Since you expressed concern about {condition}, staying alert to official warnings is most important."},
                ],
            })
            conv_id += 1

    return patterns


def generate_all_datasets():
    """Generate all datasets and save to disk."""
    os.makedirs(DATA_DIR, exist_ok=True)

    print("📊 Generating Weather QA dataset...")
    weather_qa = generate_weather_qa()
    qa_path = os.path.join(DATA_DIR, "weather_qa.json")
    with open(qa_path, "w") as f:
        json.dump(weather_qa, f, indent=2)
    print(f"   ✅ {len(weather_qa)} records → {qa_path}")

    print("📚 Generating Climate Knowledge dataset...")
    climate_knowledge = generate_climate_knowledge()
    ck_path = os.path.join(DATA_DIR, "climate_knowledge.json")
    with open(ck_path, "w") as f:
        json.dump(climate_knowledge, f, indent=2)
    print(f"   ✅ {len(climate_knowledge)} records → {ck_path}")

    print("💬 Generating Conversation Patterns dataset...")
    conversation_patterns = generate_conversation_patterns()
    cp_path = os.path.join(DATA_DIR, "conversation_patterns.json")
    with open(cp_path, "w") as f:
        json.dump(conversation_patterns, f, indent=2)
    print(f"   ✅ {len(conversation_patterns)} patterns → {cp_path}")

    print(f"\n✅ All datasets generated in {DATA_DIR}")
    return weather_qa, climate_knowledge, conversation_patterns


if __name__ == "__main__":
    generate_all_datasets()
