import pandas as pd

CSV_PATH = r"S_DB__Data_2024_태깅_이상행동.csv"
REQUIRED_SENSORS = {"IR", "RGB", "Depth", "Thermal", "Lidar"}

df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

col_filename = df.columns[0]  # 파일명
col_sensor = df.columns[1]    # 센서

grouped = df.groupby(col_filename)[col_sensor].apply(set)

from collections import defaultdict

# 빠진 센서 조합별로 파일명 그룹핑
missing_by_sensor = defaultdict(list)
for filename, sensors in grouped.items():
    missing = REQUIRED_SENSORS - sensors
    if missing:
        key = tuple(sorted(missing))
        missing_by_sensor[key].append(filename)

OUTPUT_PATH = "missing_sensors_report.txt"
lines = []
lines.append(f"전체 고유 파일명 수: {len(grouped)}")
lines.append(f"센서 누락 파일명 수: {sum(len(v) for v in missing_by_sensor.values())}")
lines.append("-" * 60)

if missing_by_sensor:
    for missing_sensors, filenames in sorted(missing_by_sensor.items()):
        sensor_str = ", ".join(missing_sensors)
        fname_str = ", ".join(sorted(filenames))
        lines.append(f"빠진 센서 [{sensor_str}]인 파일명: {fname_str}")
        lines.append("")
else:
    lines.append("모든 파일명에 5종 센서가 모두 존재합니다.")

report = "\n".join(lines)
print(report)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(report)
print(f"\n결과가 {OUTPUT_PATH}에 저장되었습니다.")