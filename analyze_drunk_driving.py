import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import seaborn as sns
import sys

# UTF-8 출력 설정
sys.stdout.reconfigure(encoding='utf-8')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
df = pd.read_csv('data/경찰청_음주운전_20241231.csv', encoding='cp949')

print("컬럼명:", df.columns.tolist())
print("\n데이터 형태:", df.shape)
print("\n처음 5개 행:")
print(df.head())
print("\n데이터 정보:")
print(df.info())

# 컬럼명 확인 후 영문으로 변경 (작업 편의성을 위해)
df.columns = ['연번', '성별', '적발횟수', '나이', '알콜농도', '측정거부', '측정일시', '관할경찰서']

print("\n변경된 컬럼명:", df.columns.tolist())
print("\n관할경찰서 샘플:")
print(df['관할경찰서'].head(20))

# 시/군 단위 지역을 광역자치단체로 매핑하는 딕셔너리
CITY_TO_PROVINCE = {
    # 경기도
    '고양': '경기도', '과천': '경기도', '광명': '경기도', '구리': '경기도', '군포': '경기도',
    '김포': '경기도', '남양주': '경기도', '동두천': '경기도', '부천': '경기도', '분당': '경기도',
    '성남': '경기도', '수원': '경기도', '시흥': '경기도', '안산': '경기도', '안성': '경기도',
    '안양': '경기도', '양주': '경기도', '양평': '경기도', '여주': '경기도', '연천': '경기도',
    '오산': '경기도', '용인': '경기도', '의왕': '경기도', '의정부': '경기도', '이천': '경기도',
    '일산': '경기도', '파주': '경기도', '평택': '경기도', '포천': '경기도', '하남': '경기도',
    '화성': '경기도', '가평': '경기도',

    # 강원도
    '강릉': '강원특별자치도', '고성': '강원특별자치도', '동해': '강원특별자치도', '삼척': '강원특별자치도',
    '속초': '강원특별자치도', '양구': '강원특별자치도', '양양': '강원특별자치도', '영월': '강원특별자치도',
    '원주': '강원특별자치도', '인제': '강원특별자치도', '정선': '강원특별자치도', '철원': '강원특별자치도',
    '춘천': '강원특별자치도', '태백': '강원특별자치도', '평창': '강원특별자치도', '홍천': '강원특별자치도',
    '화천': '강원특별자치도', '횡성': '강원특별자치도',

    # 충청북도
    '괴산': '충청북도', '단양': '충청북도', '보은': '충청북도', '영동': '충청북도', '옥천': '충청북도',
    '음성': '충청북도', '제천': '충청북도', '진천': '충청북도', '청주': '충청북도', '충주': '충청북도',
    '증평': '충청북도',

    # 충청남도
    '계룡': '충청남도', '공주': '충청남도', '금산': '충청남도', '논산': '충청남도', '당진': '충청남도',
    '보령': '충청남도', '부여': '충청남도', '서산': '충청남도', '서천': '충청남도', '아산': '충청남도',
    '예산': '충청남도', '천안': '충청남도', '청양': '충청남도', '태안': '충청남도', '홍성': '충청남도',

    # 전북특별자치도 (구 전라북도)
    '고창': '전북특별자치도', '군산': '전북특별자치도', '김제': '전북특별자치도', '남원': '전북특별자치도',
    '무주': '전북특별자치도', '부안': '전북특별자치도', '순창': '전북특별자치도', '완주': '전북특별자치도',
    '익산': '전북특별자치도', '임실': '전북특별자치도', '장수': '전북특별자치도', '전주': '전북특별자치도',
    '정읍': '전북특별자치도', '진안': '전북특별자치도',

    # 전라남도
    '강진': '전라남도', '고흥': '전라남도', '곡성': '전라남도', '광양': '전라남도', '구례': '전라남도',
    '나주': '전라남도', '담양': '전라남도', '목포': '전라남도', '무안': '전라남도', '보성': '전라남도',
    '순천': '전라남도', '신안': '전라남도', '여수': '전라남도', '영광': '전라남도', '영암': '전라남도',
    '완도': '전라남도', '장성': '전라남도', '장흥': '전라남도', '진도': '전라남도', '함평': '전라남도',
    '해남': '전라남도', '화순': '전라남도',

    # 경상북도
    '경산': '경상북도', '경주': '경상북도', '고령': '경상북도', '구미': '경상북도', '군위': '경상북도',
    '김천': '경상북도', '문경': '경상북도', '봉화': '경상북도', '상주': '경상북도', '성주': '경상북도',
    '안동': '경상북도', '영덕': '경상북도', '영양': '경상북도', '영주': '경상북도', '영천': '경상북도',
    '예천': '경상북도', '울릉': '경상북도', '울진': '경상북도', '의성': '경상북도', '청도': '경상북도',
    '청송': '경상북도', '칠곡': '경상북도', '포항': '경상북도',

    # 경상남도
    '거제': '경상남도', '거창': '경상남도', '김해': '경상남도', '남해': '경상남도', '마산': '경상남도',
    '밀양': '경상남도', '사천': '경상남도', '산청': '경상남도', '양산': '경상남도', '의령': '경상남도',
    '진주': '경상남도', '진해': '경상남도', '창녕': '경상남도', '창원': '경상남도', '통영': '경상남도',
    '하동': '경상남도', '함안': '경상남도', '함양': '경상남도', '합천': '경상남도',

    # 제주특별자치도
    '서귀포': '제주특별자치도', '제주': '제주특별자치도',
}

# 지역(도) 추출 - 경찰서명에서 도 이름 추출
def extract_province(police_station):
    """경찰서명에서 광역자치단체 이름 추출"""
    if pd.isna(police_station):
        return '기타'

    station = str(police_station)

    # 특별시/광역시 직접 매칭
    if '서울' in station:
        return '서울특별시'
    elif '부산' in station:
        return '부산광역시'
    elif '대구' in station:
        return '대구광역시'
    elif '인천' in station:
        return '인천광역시'
    elif '광주' in station:
        return '광주광역시'
    elif '대전' in station:
        return '대전광역시'
    elif '울산' in station:
        return '울산광역시'
    elif '세종' in station:
        return '세종특별자치시'

    # 도 경찰청 직접 매칭
    elif '경상남도경찰청' in station:
        return '경상남도'
    elif '전라남도경찰청' in station:
        return '전라남도'
    elif '전북특별자치도경찰청' in station:
        return '전북특별자치도'
    elif '충청남도경찰청' in station:
        return '충청남도'

    # 시/군 단위 매칭 (딕셔너리 사용)
    else:
        for city, province in CITY_TO_PROVINCE.items():
            if city in station:
                return province

    return '기타'

# 도 추출
df['도'] = df['관할경찰서'].apply(extract_province)

# 나이대 분류 (10대 단위)
def classify_age_group(age):
    """나이를 10대 단위로 분류"""
    if pd.isna(age):
        return '미상'
    age = int(age)
    if age < 20:
        return '10대'
    elif age < 30:
        return '20대'
    elif age < 40:
        return '30대'
    elif age < 50:
        return '40대'
    elif age < 60:
        return '50대'
    elif age < 70:
        return '60대'
    else:
        return '70대 이상'

df['나이대'] = df['나이'].apply(classify_age_group)

print("\n도별 분포:")
print(df['도'].value_counts())
print("\n나이대별 분포:")
print(df['나이대'].value_counts())

# 1. 지역별 음주운전 적발 통계
province_stats = df['도'].value_counts().sort_values(ascending=False)
print("\n=== 지역별 음주운전 적발 통계 ===")
print(province_stats)

# 2. 나이대별 음주운전 적발 통계
age_stats = df['나이대'].value_counts()
# 나이대 순서 정렬
age_order = ['10대', '20대', '30대', '40대', '50대', '60대', '70대 이상', '미상']
age_stats = age_stats.reindex([age for age in age_order if age in age_stats.index])
print("\n=== 나이대별 음주운전 적발 통계 ===")
print(age_stats)

# 3. 지역 x 나이대 크로스 테이블
cross_table = pd.crosstab(df['도'], df['나이대'])
cross_table = cross_table[age_stats.index]  # 나이대 순서 정렬
cross_table = cross_table.loc[province_stats.index]  # 지역을 적발 건수 순으로 정렬

print("\n=== 지역 x 나이대 크로스 테이블 ===")
print(cross_table)

# 시각화 - 각 차트를 개별 파일로 저장

# 1. 지역별 음주운전 적발 건수 (막대 그래프)
fig1, ax1 = plt.subplots(figsize=(14, 8))
province_stats.plot(kind='bar', ax=ax1, color='steelblue')
ax1.set_title('지역(도)별 음주운전 적발 건수', fontsize=18, fontweight='bold', pad=20)
ax1.set_xlabel('지역', fontsize=14)
ax1.set_ylabel('적발 건수', fontsize=14)
ax1.tick_params(axis='x', rotation=45)
ax1.grid(axis='y', alpha=0.3)

# 값 표시
for i, v in enumerate(province_stats):
    ax1.text(i, v, f'{v:,}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('1_지역별_음주운전_적발건수.png', dpi=300, bbox_inches='tight')
print("✓ '1_지역별_음주운전_적발건수.png' 저장 완료")
plt.close()

# 2. 나이대별 음주운전 적발 건수 (막대 그래프)
fig2, ax2 = plt.subplots(figsize=(12, 8))
age_stats.plot(kind='bar', ax=ax2, color='coral')
ax2.set_title('나이대별 음주운전 적발 건수', fontsize=18, fontweight='bold', pad=20)
ax2.set_xlabel('나이대', fontsize=14)
ax2.set_ylabel('적발 건수', fontsize=14)
ax2.tick_params(axis='x', rotation=45)
ax2.grid(axis='y', alpha=0.3)

# 값 표시
for i, v in enumerate(age_stats):
    ax2.text(i, v, f'{v:,}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('2_나이대별_음주운전_적발건수.png', dpi=300, bbox_inches='tight')
print("✓ '2_나이대별_음주운전_적발건수.png' 저장 완료")
plt.close()

# 3. 지역 x 나이대 히트맵
fig3, ax3 = plt.subplots(figsize=(12, 10))
sns.heatmap(cross_table, annot=True, fmt='d', cmap='YlOrRd', ax=ax3,
            cbar_kws={'label': '적발 건수'}, linewidths=0.5)
ax3.set_title('지역별 x 나이대별 음주운전 적발 건수 히트맵', fontsize=18, fontweight='bold', pad=20)
ax3.set_xlabel('나이대', fontsize=14)
ax3.set_ylabel('지역', fontsize=14)

plt.tight_layout()
plt.savefig('3_지역별_나이대별_히트맵.png', dpi=300, bbox_inches='tight')
print("✓ '3_지역별_나이대별_히트맵.png' 저장 완료")
plt.close()

# 4. 지역별 나이대 분포 (누적 막대 그래프)
fig4, ax4 = plt.subplots(figsize=(16, 8))
cross_table.plot(kind='bar', stacked=True, ax=ax4, colormap='tab10')
ax4.set_title('지역별 나이대 분포 (누적)', fontsize=18, fontweight='bold', pad=20)
ax4.set_xlabel('지역', fontsize=14)
ax4.set_ylabel('적발 건수', fontsize=14)
ax4.tick_params(axis='x', rotation=45)
ax4.legend(title='나이대', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11)
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('4_지역별_나이대_분포_누적.png', dpi=300, bbox_inches='tight')
print("✓ '4_지역별_나이대_분포_누적.png' 저장 완료")
plt.close()

print("\n모든 시각화 파일이 개별적으로 저장되었습니다!")

# 추가 분석: 가장 많은 적발이 발생한 지역-나이대 조합 찾기
print("\n=== TOP 10 지역-나이대 조합 ===")
cross_table_melted = cross_table.reset_index().melt(id_vars='도', var_name='나이대', value_name='적발건수')
top_combinations = cross_table_melted.nlargest(10, '적발건수')
for idx, row in top_combinations.iterrows():
    print(f"{row['도']:12s} - {row['나이대']:8s}: {row['적발건수']:6,}건")
