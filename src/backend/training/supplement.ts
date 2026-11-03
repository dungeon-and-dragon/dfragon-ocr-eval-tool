import { createHash } from 'node:crypto'
import { readFile, realpath, stat } from 'node:fs/promises'
import { join, isAbsolute, relative } from 'node:path'
import {
  LEGACY_SUPPLEMENT_LAYOUT,
  characterDistribution,
  type SupplementInfo,
  type SupplementOptions,
  type SupplementPlan
} from '../../shared/supplement'
import { isRecord } from '../evaluation/validation'
import { parseDataset } from './library'
import { isSupplementPlan, parseSupplementOptions } from './supplement-validation'

export async function supplementInfo(root: string): Promise<SupplementInfo> {
  const bytes = await readFile(join(root, 'dataset.json'))
  const ready: unknown = JSON.parse(await readFile(join(root, 'ready.json'), 'utf8'))
  if (
    !isRecord(ready) ||
    ready.datasetSha256 !== createHash('sha256').update(bytes).digest('hex')
  ) {
    throw new Error('저장된 실제 데이터가 변경되었습니다.')
  }
  const dataset = parseDataset(JSON.parse(bytes.toString('utf8')))
  const dictionary = new Set(
    (await readFile(join(root, 'model/characters.txt'), 'utf8')).trimEnd().split(/\r?\n/)
  )
  dictionary.add(' ')
  const train = dataset.samples.filter((row) => row.split === 'train' && row.text !== null)
  const all = characterDistribution(
    dataset.samples.flatMap((row) => (row.text === null ? [] : [row.text]))
  )
  return {
    real: characterDistribution(train.map((row) => row.text!)),
    missing: all.frequencies.filter((item) => !dictionary.has(item.character)),
    dictionarySize: dictionary.size - 1,
    width: [...train].sort((a, b) => a.width - b.width)[Math.floor(train.length / 2)]?.width ?? 160,
    height:
      [...train].sort((a, b) => a.height - b.height)[Math.floor(train.length / 2)]?.height ?? 32
  }
}

export async function readSupplement(
  root: string,
  directory: string,
  examples: boolean
): Promise<{
  plan: SupplementPlan
  options: SupplementOptions
  examples: { text: string; image: string }[]
  sha256: string
}> {
  const file = join(directory, 'supplement.json')
  if ((await stat(file)).size > 256 * 1024 * 1024) {
    throw new Error('합성 결과 파일이 너무 큽니다.')
  }
  const bytes = await readFile(file)
  const value: unknown = JSON.parse(bytes.toString('utf8'))
  if (
    !isRecord(value) ||
    (value.schemaVersion !== 1 && value.schemaVersion !== 2) ||
    !isRecord(value.options) ||
    !isSupplementPlan(value.plan) ||
    !Array.isArray(value.samples)
  ) {
    throw new Error('합성 결과 형식이 올바르지 않습니다.')
  }
  const savedOptions =
    value.schemaVersion === 1 ? { ...value.options, ...LEGACY_SUPPLEMENT_LAYOUT } : value.options
  const result = {
    plan: value.plan,
    options: parseSupplementOptions(savedOptions),
    examples: [] as { text: string; image: string }[],
    sha256: createHash('sha256').update(bytes).digest('hex')
  }
  if (examples) {
    for (const row of value.samples.slice(0, 6)) {
      if (!isRecord(row) || typeof row.text !== 'string' || typeof row.image !== 'string') {
        throw new Error('합성 예시가 올바르지 않습니다.')
      }
      const path = await realpath(join(root, row.image))
      const rel = relative(join(directory, 'images'), path)
      if (
        isAbsolute(rel) ||
        !/^\d{6}\.png$/.test(rel) ||
        (await stat(path)).size > 16 * 1024 * 1024
      ) {
        throw new Error('합성 예시 경로가 올바르지 않습니다.')
      }
      const image = await readFile(path)
      if (createHash('sha256').update(image).digest('hex') !== row.sha256) {
        throw new Error('합성 예시가 변경되었습니다.')
      }
      result.examples.push({
        text: row.text,
        image: `data:image/png;base64,${image.toString('base64')}`
      })
    }
  }
  return result
}
