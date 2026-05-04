// Import CSV as text
const demoCowsCSV = `cow_L,cow_R,cow_L_URL,cow_R_URL,pair_id,question_type,expert_answer,question_num,HIT,question_id,severity_L,severity_R
5087,7128,https://object-arbutus.cloud.computecanada.ca/awp-public-compressed-20260109-8427/ubc/good/280521_pm_MVI_0180_short_27_5087.MP4,https://object-arbutus.cloud.computecanada.ca/awp-public-compressed-20260109-8427/ubc/good/010421_MVI_0114_short_18_7128.MP4,500.0,,-1.0,q_new1,0,0-q_new1,moderate,mild
7045,5087,https://object-arbutus.cloud.computecanada.ca/awp-public-compressed-20260109-8427/ubc/good/130521_am_MVI_0162_short_30_7045.MP4,https://object-arbutus.cloud.computecanada.ca/awp-public-compressed-20260109-8427/ubc/good/280521_pm_MVI_0180_short_27_5087.MP4,190.0,pos_attention_easy,2.0,q6,0,0-q6,healthy,moderate
6088,5118,https://object-arbutus.cloud.computecanada.ca/awp-public-compressed-20260109-8427/ubc/good/040621_pm_MVI_0187_short_11_6088.MP4,https://object-arbutus.cloud.computecanada.ca/awp-public-compressed-20260109-8427/ubc/good/070521_pm_MVI_0159_short_22_5118.MP4,133.0,,-1.5,q1,0,0-q1,severe,healthy
5087,5087,https://object-arbutus.cloud.computecanada.ca/awp-public-compressed-20260109-8427/ubc/good/280521_pm_MVI_0180_short_27_5087.MP4,https://object-arbutus.cloud.computecanada.ca/awp-public-compressed-20260109-8427/ubc/good/280521_pm_MVI_0180_short_27_5087.MP4,-3.0,neg_attention,0.0,q2,0,0-q2,moderate,moderate
6046,6088,https://object-arbutus.cloud.computecanada.ca/awp-public-compressed-20260109-8427/ubc/good/040621_pm_MVI_0188_short_20_6046.MP4,https://object-arbutus.cloud.computecanada.ca/awp-public-compressed-20260109-8427/ubc/good/040621_pm_MVI_0187_short_11_6088.MP4,29.0,,1.5,q3,0,0-q3,mild,severe
7136,4035,https://object-arbutus.cloud.computecanada.ca/awp-public-compressed-20260109-8427/ubc/good/110621_am_MVI_0193_short_23_7136.MP4,https://object-arbutus.cloud.computecanada.ca/awp-public-compressed-20260109-8427/ubc/bad/280521_am_MVI_0178_short_47_4035.MP4,129.0,,2.0,q4,0,0-q4,healthy,moderate
6088,7163,https://object-arbutus.cloud.computecanada.ca/awp-public-compressed-20260109-8427/ubc/good/040621_pm_MVI_0187_short_11_6088.MP4,https://object-arbutus.cloud.computecanada.ca/awp-public-compressed-20260109-8427/ubc/good/210521_pm_MVI_0174_short_24_7163.MP4,82.0,,-1.5,q5,0,0-q5,severe,mild
5087,6096,https://object-arbutus.cloud.computecanada.ca/awp-public-compressed-20260109-8427/ubc/good/280521_pm_MVI_0180_short_27_5087.MP4,https://object-arbutus.cloud.computecanada.ca/awp-public-compressed-20260109-8427/ubc/good/300421_pm_MVI_0157_short_46_6096.MP4,213.0,pos_attention_easy,-2.0,q7,0,0-q7,moderate,severe
7128,4035,https://object-arbutus.cloud.computecanada.ca/awp-public-compressed-20260109-8427/ubc/good/010421_MVI_0114_short_18_7128.MP4,https://object-arbutus.cloud.computecanada.ca/awp-public-compressed-20260109-8427/ubc/bad/280521_am_MVI_0178_short_47_4035.MP4,43.0,pos_attention_easy,2.0,q8,0,0-q8,mild,moderate
6025,6088,http://object-arbutus.cloud.computecanada.ca/ubc_farm_2022_compressed/compressed_010421_MVI_0114_short_12_6025.MP4,https://object-arbutus.cloud.computecanada.ca/awp-public-compressed-20260109-8427/ubc/good/040621_pm_MVI_0187_short_11_6088.MP4,9.0,,1.5,q9,0,0-q9,healthy,severe
7128,6088,https://object-arbutus.cloud.computecanada.ca/awp-public-compressed-20260109-8427/ubc/good/010421_MVI_0114_short_18_7128.MP4,https://object-arbutus.cloud.computecanada.ca/awp-public-compressed-20260109-8427/ubc/good/040621_pm_MVI_0187_short_11_6088.MP4,331.0,,1.5,q10,0,0-q10,mild,severe
5087,6094,https://object-arbutus.cloud.computecanada.ca/awp-public-compressed-20260109-8427/ubc/good/280521_pm_MVI_0180_short_27_5087.MP4,https://object-arbutus.cloud.computecanada.ca/awp-public-compressed-20260109-8427/ubc/good/040621_pm_MVI_0189_short_34_6094.MP4,111.0,,-2.0,q11,0,0-q11,moderate,healthy
5087,6046,https://object-arbutus.cloud.computecanada.ca/awp-public-compressed-20260109-8427/ubc/good/280521_pm_MVI_0180_short_27_5087.MP4,https://object-arbutus.cloud.computecanada.ca/awp-public-compressed-20260109-8427/ubc/good/040621_pm_MVI_0188_short_20_6046.MP4,392.0,,-2.0,q12,0,0-q12,moderate,mild`

export interface DemoPair {
  cow_L: string
  cow_R: string
  cow_L_URL: string
  cow_R_URL: string
  expert_answer: number
  question_type: string
  question_num: string
  severity_L: string
  severity_R: string
}

export interface DemoCow {
  id: string
  name: string
  videoUrl: string
  severity: string
}

let parsedDemoData: DemoPair[] | null = null
let demoCows: DemoCow[] | null = null

export function resetDemoCache() {
  parsedDemoData = null
  demoCows = null
}

export function parseDemoCSV(): DemoPair[] {
  // Use cached data if available
  if (parsedDemoData) return parsedDemoData

  const lines = demoCowsCSV.trim().split('\n')
  
  parsedDemoData = lines.slice(1).map(line => {
    const values = line.split(',')
    return {
      cow_L: values[0].trim(),
      cow_R: values[1].trim(),
      cow_L_URL: values[2].trim(),
      cow_R_URL: values[3].trim(),
      expert_answer: parseFloat(values[6]) || 0,
      question_type: values[5]?.trim() || '',
      question_num: values[7]?.trim() || '',
      severity_L: values[10]?.trim() || 'unknown',
      severity_R: values[11]?.trim() || 'unknown'
    }
  })

  return parsedDemoData
}

export function getDemoCows(): DemoCow[] {
  if (demoCows) return demoCows

  const pairs = parseDemoCSV()
  const cowMap = new Map<string, DemoCow>()

  pairs.forEach(pair => {
    if (!cowMap.has(pair.cow_L)) {
      cowMap.set(pair.cow_L, {
        id: pair.cow_L,
        name: `Cow ${pair.cow_L}`,
        videoUrl: pair.cow_L_URL,
        severity: pair.severity_L
      })
    }
    if (!cowMap.has(pair.cow_R)) {
      cowMap.set(pair.cow_R, {
        id: pair.cow_R,
        name: `Cow ${pair.cow_R}`,
        videoUrl: pair.cow_R_URL,
        severity: pair.severity_R
      })
    }
  })

  demoCows = Array.from(cowMap.values())
  return demoCows
}

export function getRandomDemoPair(): DemoPair | null {
  const pairs = parseDemoCSV()
  if (pairs.length === 0) return null
  return pairs[Math.floor(Math.random() * pairs.length)]
}

export function getDemoPairByIndex(index: number): DemoPair | null {
  const pairs = parseDemoCSV()
  if (index < 0 || index >= pairs.length) return null
  return pairs[index]
}
