import type { SupplementOptions, SupplementPreview, SupplementInfo } from './supplement'
import type { EvaluationSample, Metrics, Result } from './contracts'

export interface LibraryModel {
  id: string
  name: string
  preset: 'korean-ppocrv5'
  kind: 'pretrained' | 'finetuned' | 'expanded'
  parentId: string | null
  registeredAt: string
  files: {
    name: 'weights.pdparams' | 'characters.txt' | 'evaluation.json'
    bytes: number
    sha256: string
  }[]
}

export interface LibrarySample {
  id: string
  captureId: string
  slot: number
  width: number
  height: number
  text: string | null
  excluded: boolean
  split: 'train' | 'val' | 'test' | 'unassigned'
}

// dfragon main cuts raid name clusters at 4/17 of the height; dfragon #1152 proposes 6/17.
export const NICKNAME_DETACHED_GAPS = ['4/17', '6/17'] as const

export interface TrainingOptions {
  additionalCharacters?: string
  supplementId?: string
  nicknameDetachedGap?: (typeof NICKNAME_DETACHED_GAPS)[number]
  pythonExecutable: string
  upstreamDirectory: string
  epochs: number
  batchSize: number
  learningRate: number
}

export interface TrainingSnapshot {
  status:
    | 'idle'
    | 'downloading'
    | 'preparing'
    | 'ready'
    | 'training'
    | 'evaluating'
    | 'cancelling'
    | 'completed'
    | 'cancelled'
    | 'failed'
    | 'publishing'
  message: string
  error: string | null
  directory: string | null
  counts: { train: number; val: number; test: number; skipped: number } | null
  model: LibraryModel | null
  epoch: number
  epochs: number
  metrics: Metrics | null
  samples: EvaluationSample[]
  publishedModelId: string | null
  supplement?: SupplementPreview | null
  addedCharacters?: string[]
}

export interface TrainingApi {
  login(): Promise<Result<null>>
  logout(): Promise<Result<null>>
  models(): Promise<Result<LibraryModel[]>>
  defaults(): Promise<Result<Partial<TrainingOptions>>>
  download(modelId: string, directory: string): Promise<Result<null>>
  open(directory: string): Promise<Result<null>>
  supplementInfo(): Promise<Result<SupplementInfo>>
  previewSupplement(options: SupplementOptions): Promise<Result<SupplementPreview>>
  start(options: TrainingOptions): Promise<Result<null>>
  cancel(): Promise<Result<null>>
  publish(name: string): Promise<Result<LibraryModel>>
  readImage(path: string): Promise<Result<string>>
  getSnapshot(): Promise<Result<TrainingSnapshot>>
  onSnapshot(listener: (snapshot: TrainingSnapshot) => void): () => void
}

export const TRAINING_IPC = {
  login: 'training:login',
  logout: 'training:logout',
  models: 'training:models',
  defaults: 'training:defaults',
  download: 'training:download',
  open: 'training:open',
  start: 'training:start',
  supplementInfo: 'training:supplement-info',
  previewSupplement: 'training:preview-supplement',
  cancel: 'training:cancel',
  publish: 'training:publish',
  readImage: 'training:read-image',
  getSnapshot: 'training:get-snapshot',
  snapshot: 'training:snapshot'
} as const
