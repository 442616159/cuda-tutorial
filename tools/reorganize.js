/**
 * reorganize.js —— 一次性整理脚本（用 Node 执行）
 *
 * 作用：
 *   1. 把 code/chXX_yyy/ 移到 code/lessons/chXX_zzz/
 *   2. 把 ch08 的三个项目移到 code/projects/
 *   3. 顺带修正所有 .cu / .h / .md 里的 #include 与相对路径
 *
 * 用法（在 CUDALearning 目录下）：
 *   node tools/reorganize.js
 */
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const CODE = path.join(ROOT, 'code');

// ---------- 目录映射：旧 -> 新（相对 code/）----------
const DIR_MAP = {
  // 第 0 章
  'ch00_hello':                 'lessons/ch00_hello',
  // 第 1 章
  'ch01_vector_add':            'lessons/ch01_indices',
  // 第 2 章（8 个目录合并成一个）
  'ch02_matrix_mul':            'lessons/ch02_memory',
  // 第 3 章
  'ch03_sync':                  'lessons/ch03_sync/barrier',
  'ch03_syncwarp':              'lessons/ch03_sync/warp_sync',
  'ch03_threadfence':           'lessons/ch03_sync/fence',
  'ch03_shuffle_reduce':        'lessons/ch03_sync/shuffle',
  'ch03_vote':                  'lessons/ch03_sync/vote',
  'ch03_atomics':               'lessons/ch03_atomics',
  'ch03_histogram':             'lessons/ch03_atomics_histogram',
  'ch03_reduce_cg':             'lessons/ch03_cooperative',
  'ch03_streams':               'lessons/ch03_streams',
  'ch03_graphs':                'lessons/ch03_graphs',
  // 第 4 章
  'ch04_timer':                 'lessons/ch04_timing/timer',
  'ch04_bandwidth_test':        'lessons/ch04_timing/bandwidth',
  'ch04_roofline':              'lessons/ch04_perf_model',
  'ch04_occupancy':             'lessons/ch04_occupancy',
  'ch04_matmul_opt':            'lessons/ch04_case_study',
  'ch04_sanitizer_bugs':        'lessons/ch04_debugging',
  // 第 5 章
  'ch05_reduce':                'lessons/ch05_reduce',
  'ch05_scan':                  'lessons/ch05_scan',
  'ch05_stencil':               'lessons/ch05_stencil',
  'ch05_sort':                  'lessons/ch05_sort',
  'ch05_spmv':                  'lessons/ch05_spmv',
  'ch05_bfs':                   'lessons/ch05_graph',
  'ch05_numeric':               'lessons/ch05_numeric',
  'ch05_cg':                    'lessons/ch05_cg',
  // 第 6 章
  'ch06_dynparallel':           'lessons/ch06_advanced/dynparallel',
  'ch06_wmma':                  'lessons/ch06_advanced/wmma',
  'ch06_cpasync':               'lessons/ch06_advanced/cpasync',
  'ch06_cluster':               'lessons/ch06_advanced/cluster',
  'ch06_unified':               'lessons/ch06_advanced/unified',
  'ch06_vmm':                   'lessons/ch06_advanced/vmm',
  'ch06_multigpu':              'lessons/ch06_advanced/multigpu',
  'ch06_graphs':                'lessons/ch06_advanced/graphs_advanced',
  // 第 7 章（每个库一个子目录）
  'ch07_cublas':                'lessons/ch07_ecosystem/cublas',
  'ch07_cufft':                 'lessons/ch07_ecosystem/cufft',
  'ch07_curand':                'lessons/ch07_ecosystem/curand',
  'ch07_thrust':                'lessons/ch07_ecosystem/thrust',
  'ch07_cub':                   'lessons/ch07_ecosystem/cub',
  'ch07_cusparse':              'lessons/ch07_ecosystem/cusparse',
  'ch07_cudnn':                 'lessons/ch07_ecosystem/cudnn',
  'ch07_cmake':                 'projects/cmake_skeleton',
  'ch07_python':                'lessons/ch07_python',
  // 第 8 章项目
  'ch08_project_sgemm':         'projects/sgemm',
  'ch08_project_raytracer':     'projects/raytracer',
  'ch08_project_nbody':         'projects/nbody',
};

// ---------- 路径引用改写规则 ----------
// 旧写法 -> 新写法。注意顺序：先长后短，避免前缀误伤。
const REF_RULES = [
  // 顶层 include（code/lessons/xxx/*.cu）
  ['"../common/', '"../../common/'],
  ['../common/cuda_check.h', '../common/cuda_check.h'], // 占位，稍后按深度统一处理
];

/** 计算某个文件相对 code/common 需要几层 ../ */
function depthToCommon(relFromCode) {
  // relFromCode 形如 lessons/ch02_memory/matrix_mul.cu
  const depth = relFromCode.split('/').length - 1; // 去掉文件名
  return '../'.repeat(depth) + 'common/';
}

/** 改写单个文本文件里的路径引用 */
function rewrite(text, newRel) {
  const prefixOld = '../common/';
  const prefixNew = depthToCommon(newRel);

  let out = text;

  // 1) 修正 #include "../common/xxx"
  out = out.split(prefixOld).join(prefixNew);

  // 2) 修正注释/文档里出现的旧目录名
  for (const [oldDir, newDir] of Object.entries(DIR_MAP)) {
    // 只替换 "code/旧目录" 或裸的 "旧目录/" 形式
    out = out.split('code/' + oldDir + '/').join('code/' + newDir + '/');
  }
  return out;
}

function walk(dir, cb) {
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) walk(p, cb);
    else cb(p);
  }
}

function moveDir(src, dst) {
  if (!fs.existsSync(src)) {
    console.log(`  [跳过] 源不存在: ${path.relative(ROOT, src)}`);
    return false;
  }
  fs.mkdirSync(path.dirname(dst), { recursive: true });
  fs.renameSync(src, dst);
  console.log(`  移动 ${path.relative(CODE, src)}  ->  ${path.relative(CODE, dst)}`);
  return true;
}

console.log('=== 第 1 步：移动目录 ===');
for (const [oldDir, newDir] of Object.entries(DIR_MAP)) {
  moveDir(path.join(CODE, oldDir), path.join(CODE, newDir));
}

console.log('\n=== 第 2 步：清理空目录 ===');
function removeEmpty(dir) {
  if (!fs.existsSync(dir)) return;
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    if (e.isDirectory()) removeEmpty(path.join(dir, e.name));
  }
  if (dir !== CODE && fs.readdirSync(dir).length === 0) {
    fs.rmdirSync(dir);
    console.log(`  删除空目录 ${path.relative(CODE, dir)}`);
  }
}
removeEmpty(CODE);

console.log('\n=== 第 3 步：改写代码里的路径引用 ===');
const EXTS = ['.cu', '.h', '.cuh', '.cpp', '.py', '.md', '.sh', '.txt', '.cmake'];
walk(CODE, (file) => {
  const ext = path.extname(file);
  if (!EXTS.includes(ext) && path.basename(file) !== 'CMakeLists.txt') return;
  const rel = path.relative(CODE, file).split(path.sep).join('/');
  const before = fs.readFileSync(file, 'utf8');
  const after = rewrite(before, rel);
  if (after !== before) {
    fs.writeFileSync(file, after, 'utf8');
    console.log(`  更新 ${rel}`);
  }
});

console.log('\n=== 完成 ===');
