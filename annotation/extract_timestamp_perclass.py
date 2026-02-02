import pandas as pd

def time_to_sec(t_str):
    if pd.isna(t_str) or t_str == '-': return None
    m, s = map(int, str(t_str).split(':'))
    return m * 60 + s

def sec_to_time(s):
    return f"{s // 60}:{s % 60:02d}"

# 데이터 로드 및 RGB 필터링
df = pd.read_csv('ETRI_Data_2024_태깅_이상행동.csv')
rgb_df = df[df['센서'] == 'RGB'].copy()

# 이벤트별 컬럼 매핑 (사람 등장/퇴장 중복 처리 반영)
events = {
    '도난': ('사람 등장', '도난_시작', '도난_종료'),
    '싸움': ('사람 등장.1', '싸움_시작', '싸움_종료'),
    '배회': ('사람 등장.2', '배회_시작', '배회_종료'),
    '침입': ('사람 등장.3', '침입_시작', '침입_종료'),
    '쓰러짐': ('사람 등장.4', '쓰러짐_시작', '쓰러짐_종료'),
    '유기': ('사람 등장.5', '유기_시작', '유기_종료')
}

for name, cols in events.items():
    enter_col, start_col, end_col = cols
    results = []
    
    for _, row in rgb_df.iterrows():
        t_ent = time_to_sec(row[enter_col])
        t_st  = time_to_sec(row[start_col])
        t_en  = time_to_sec(row[end_col])
        
        if None in [t_ent, t_st, t_en]: continue
        
        results.append({
            'file_name': f"{row['파일명']}_RGB_{name}_전체",
            'start_time': sec_to_time(t_st - t_ent),
            'end_time': sec_to_time(t_en - t_ent),
            'duration': sec_to_time(t_en - t_st)
        })
    
    # CSV 저장
    pd.DataFrame(results).to_csv(f'{name}_timestamp.csv', index=False, encoding='utf-8-sig')