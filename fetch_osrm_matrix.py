"""One-time OSRM driving matrix fetch for the 10 Cagayan de Oro TSP vertices.

Fill landmark_coordinates.csv using verified OSM map pins (lat/lon).
Run: pip install requests
     python fetch_osrm_matrix.py
Public OSRM demo is best-effort; do not repeatedly poll it.
"""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import requests

BASE = Path(__file__).resolve().parent
INPUT = BASE / 'landmark_coordinates.csv'
OUTPUT = BASE / 'distance_matrix.csv'
METADATA = BASE / 'distance_matrix_source.json'
ENDPOINT = 'https://router.project-osrm.org/table/v1/driving/'


def main():
    with INPUT.open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 10 or [r['vertex'] for r in rows] != [f'V{i}' for i in range(10)]:
        raise ValueError('CSV must have exactly V0..V9 in that order.')
    try:
        coords = [(float(r['latitude']), float(r['longitude'])) for r in rows]
    except (TypeError, ValueError) as exc:
        raise ValueError('Fill in each latitude and longitude with verified map-pin coordinates.') from exc
    if any(not (8.35 < lat < 8.65 and 124.45 < lon < 124.85) for lat, lon in coords):
        raise ValueError('At least one pin is outside the approximate Cagayan de Oro vicinity. Check coordinates.')
    coordinate_string = ';'.join(f'{lon:.7f},{lat:.7f}' for lat, lon in coords)
    url = ENDPOINT + coordinate_string
    headers = {'User-Agent': 'CagayanDeOroStudentTSP/1.0 (one-time academic OSRM matrix request)'}
    response = requests.get(url, params={'annotations': 'distance'}, headers=headers, timeout=90)
    response.raise_for_status()
    data = response.json()
    if data.get('code') != 'Ok':
        raise RuntimeError(f'OSRM routing error: {data.get("message", data.get("code"))}')
    distances = data.get('distances')
    if not distances or len(distances) != 10 or any(len(r) != 10 for r in distances):
        raise ValueError('API did not return a complete 10x10 distance matrix.')
    if any(distances[i][j] is None for i in range(10) for j in range(10) if i != j):
        raise ValueError('Unreachable origin/destination pair; inspect pins and road access.')
    # OSRM snaps each pin to the navigable road. Warn if substantially displaced.
    for side in ('sources', 'destinations'):
        for i, waypoint in enumerate(data.get(side, [])):
            snap = waypoint.get('distance', 0)
            if snap > 250:
                raise ValueError(f'{rows[i]["vertex"]} is {snap:.0f}m from its snapped {side} road. Verify pin.')
    # Preserve meter precision in km; do not round before optimization.
    km = [[0.0 if i == j else distances[i][j] / 1000.0 for j in range(10)] for i in range(10)]
    with OUTPUT.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['from\\to'] + [r['vertex'] for r in rows])
        for i, row in enumerate(km):
            w.writerow([f'V{i}'] + [f'{v:.6f}' for v in row])
    metadata = {
        'source': 'OpenStreetMap road data routed via public OSRM demo Table API',
        'endpoint': 'https://router.project-osrm.org/table/v1/driving/',
        'request_utc': datetime.now(timezone.utc).isoformat(),
        'routing_mode': 'driving',
        'metric': 'distance in km along fastest OSRM driving routes',
        'notes': 'One-off snapshot. Map network may change. Free public demo: no service guarantees.',
        'coordinates': [{**{'vertex': r['vertex'], 'name': r['name']}, 'latitude': lat, 'longitude': lon} for r, (lat, lon) in zip(rows, coords)],
    }
    METADATA.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('Created:', OUTPUT)
    print('Created:', METADATA)
    print('Driving distance matrix (km):')
    for row in km:
        print(' '.join(f'{d:7.3f}' for d in row))


if __name__ == '__main__':
    main()
