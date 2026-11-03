import { createHash } from 'node:crypto'
import { mkdtemp, readFile, readdir, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, expect, test } from 'vitest'
import { LibraryClient, parseDataset, parseModel } from './library'
import { parseTrainingOptions } from './service'

const id = '00000000-0000-4000-8000-000000000001'
const contents = new Map([
  ['weights.pdparams', Buffer.from('fake weights')],
  ['characters.txt', Buffer.from('가\n나\n')]
])
const model = {
  id,
  name: '합성 모델',
  preset: 'korean-ppocrv5',
  kind: 'pretrained',
  parentId: null,
  registeredAt: '2026-09-27T00:00:00Z',
  files: [...contents].map(([name, bytes]) => ({
    name,
    bytes: bytes.length,
    sha256: createHash('sha256').update(bytes).digest('hex')
  }))
}
const samples = ['train', 'val', 'test', 'unassigned'].map((split, index) => ({
  id: `${id}-${index + 1}`,
  captureId: id,
  slot: index + 1,
  width: 4,
  height: 2,
  text: '가',
  excluded: false,
  split
}))
const dataset = { exportedAt: '2026-09-27T00:00:00Z', samples }
const directories: string[] = []
afterEach(async () => {
  for (const directory of directories.splice(0)) {
    await rm(directory, { recursive: true, force: true })
  }
})

test('valid server models retain file hashes; duplicate names and path injection fail', () => {
  expect(parseModel(model)).toEqual(model)
  expect(() => parseModel({ ...model, files: [...model.files, model.files[0]] })).toThrow()
  expect(() => parseModel({ ...model, id: '../outside' })).toThrow()
  expect(() =>
    parseModel({ ...model, files: [{ ...model.files[0], name: '../weights' }, model.files[1]] })
  ).toThrow()
  expect(() => parseDataset({ ...dataset, samples: [...samples, samples[0]] })).toThrow()
})

test('download filters excluded/unassigned rows, preserves server assignments and completes only after all files', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'ocr-library-'))
  directories.push(directory)
  const requested: string[] = []
  const png = Buffer.alloc(24)
  Buffer.from('89504e470d0a1a0a', 'hex').copy(png)
  png.writeUInt32BE(4, 16)
  png.writeUInt32BE(2, 20)
  const request: typeof fetch = async (input) => {
    const path = new URL(String(input)).pathname
    requested.push(path)
    if (path === `/api/models/${id}`) {
      return Response.json(model)
    }
    if (path === '/api/export/manifest') {
      return Response.json(dataset)
    }
    if (path.includes('/files/')) {
      return new Response(contents.get(path.split('/').at(-1)!)!)
    }
    return new Response(png)
  }
  const result = await new LibraryClient(request).download(
    id,
    directory,
    new AbortController().signal,
    () => undefined
  )
  expect(result.counts).toEqual({ train: 1, val: 1, test: 1, skipped: 1 })
  expect(requested.some((path) => path.includes(`${id}-4`))).toBe(false)
  const saved = JSON.parse(await readFile(join(result.folder, 'dataset.json'), 'utf8'))
  expect(saved.samples.map((row: { split: string }) => row.split)).toEqual(['train', 'val', 'test'])
  expect(
    JSON.parse(await readFile(join(result.folder, 'ready.json'), 'utf8')).datasetSha256
  ).toHaveLength(64)
})

test('hash mismatch never produces a ready snapshot', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'ocr-library-failed-'))
  directories.push(directory)
  const request: typeof fetch = async (input) => {
    const path = new URL(String(input)).pathname
    if (path === `/api/models/${id}`) {
      return Response.json(model)
    }
    if (path === '/api/export/manifest') {
      return Response.json(dataset)
    }
    return new Response('different weights')
  }
  await expect(
    new LibraryClient(request).download(
      id,
      directory,
      new AbortController().signal,
      () => undefined
    )
  ).rejects.toThrow()
  const [experiment] = await readdir(directory)
  expect(await readdir(join(directory, experiment))).not.toContain('ready.json')
})

test('invalid training settings fail before a process is launched', () => {
  const options = {
    pythonExecutable: join(tmpdir(), 'python.exe'),
    upstreamDirectory: join(tmpdir(), 'PaddleOCR'),
    epochs: 1,
    batchSize: 2,
    learningRate: 0.00001
  }
  expect(parseTrainingOptions(options)).toEqual({
    ...options,
    additionalCharacters: '',
    nicknameDetachedGap: '4/17'
  })
  expect(parseTrainingOptions({ ...options, nicknameDetachedGap: '6/17' })).toMatchObject({
    nicknameDetachedGap: '6/17'
  })
  for (const patch of [
    { nicknameDetachedGap: '5/17' },
    { epochs: 0 },
    { batchSize: 129 },
    { learningRate: NaN },
    { learningRate: 0 }
  ]) {
    expect(() => parseTrainingOptions({ ...options, ...patch })).toThrow()
  }
})
