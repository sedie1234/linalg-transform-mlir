// code-highlight.js — 의존성 0 자체 구현 신택스 하이라이터 (MLIR + C++)
// 학습 워크스페이스 공용. <pre> / <pre><code> 블록을 토큰 분리해 설명(prose)과
// 실제 코드를 시각적으로 구분한다. linalg/scf/transform 등 API 식별자는 .tok-api 로 강조.
// 자체 CSS 를 <head> 에 주입하므로 self-contained 문서(카탈로그·docs)도 include 한 줄이면 동작.

(function () {
  'use strict';

  var API_NAMES = [
    'generalizeNamedOp', 'specializeGenericOp', 'fuseElementwiseOps',
    'areElementwiseOpsFusable', 'interchangeGenericOp', 'promoteSubViews',
    'promoteSubviewsPrecondition', 'vectorize', 'vectorizeOpPrecondition',
    'splitReduction', 'splitReductionByScaling', 'splitOp', 'peelLoop',
    'tileUsingSCF', 'tileLinalgOp', 'tileConsumerAndFuseProducersUsingSCF',
    'tileAndFuseProducerOfSlice', 'tileToForallOpUsingTileSizes',
    'tileReductionUsingScf', 'tileReductionUsingForall',
    'makeTiledLoopRanges', 'makeTiledShapes', 'transformIndexOps',
    'rewriteAsPaddedOp', 'pack', 'packMatmulGreedily', 'lowerPack', 'lowerUnPack',
    'rewriteInIm2Col', 'winogradConv2D', 'inferConvolutionDims',
    'collapseOpIterationDims', 'areDimSequencesPreserved', 'dropUnitDims',
    'bufferizeToAllocation', 'hoistRedundantVectorTransfers',
    'hoistRedundantVectorBroadcasts', 'hoistLoopInvariantSubsets',
    'linalgOpToLoops', 'linalgOpToAffineLoops', 'linalgOpToParallelLoops',
    'loopUnrollByFactor', 'blockPackMatmul',
    'applyPatternsAndFoldGreedily', 'applyPartialConversion', 'applyFullConversion',
    'getIndexingMapsArray', 'getIteratorTypesArray', 'getStaticLoopRanges',
    'getDpsInputs', 'getDpsInits', 'inlineRegionBefore', 'replaceOp',
    'notifyMatchFailure', 'hasPureTensorSemantics', 'hasPureBufferSemantics'
  ];
  var API_BARE = new RegExp('\\b(' + API_NAMES.join('|') + ')\\b', 'g');
  var API_POP = /\bpopulate[A-Za-z]+\b/g;
  var API_NS = /\b(mlir|linalg|scf|tensor|transform|affine|memref|vector|arith|bufferization|func|math)::([A-Za-z_]\w*)/g;

  // placeholder = \x01 + idx + \x01 (control char 래핑 — 코드에 등장하지 않고,
  // 숫자 정규식은 (?<!\x01) lookbehind 로, 그 외 정규식은 패턴상 placeholder 를 안 건드림)
  var SOH = '\x01';
  function maskFactory(tokens) {
    return function (m, cls) {
      var i = tokens.length;
      tokens.push('<span class="tok-' + cls + '">' + m + '</span>');
      return SOH + i + SOH;
    };
  }
  function restore(s, tokens) {
    for (var g = 0; g < 60 && s.indexOf(SOH) !== -1; g++) {
      s = s.replace(/\x01(\d+)\x01/g, function (_, i) { return tokens[+i]; });
    }
    return s;
  }

  function highlightMLIR(code) {
    var tokens = []; var mask = maskFactory(tokens); var s = code;
    s = s.replace(/\/\*[\s\S]*?\*\//g, function (m) { return mask(m, 'com'); });
    s = s.replace(/\/\/[^\n]*/g, function (m) { return mask(m, 'com'); });
    s = s.replace(/&quot;[^&]*&quot;/g, function (m) { return mask(m, 'str'); });
    s = s.replace(/"[^"]*"/g, function (m) { return mask(m, 'str'); });
    s = s.replace(/\b[a-z][\w]*\.[a-zA-Z_][\w.]*/g, function (m) { return mask(m, 'op'); });
    s = s.replace(/\b(memref|tensor|vector|i1|i8|i16|i32|i64|i128|f16|bf16|f32|f64|f80|index|none|complex)\b/g, function (m) { return mask(m, 'ty'); });
    s = s.replace(/\b(func|module|to|step|iter_args|ins|outs|reduce|yield|return|else|affine_map|affine_set|integer_set|dimensions|permutation|min|max|floordiv|ceildiv|mod|true|false|dense|loc)\b/g, function (m) { return mask(m, 'kw'); });
    s = s.replace(/%[\w$.#-]+/g, function (m) { return mask(m, 'val'); });
    s = s.replace(/#[\w$.<>-]*/g, function (m) { return mask(m, 'attr'); });
    s = s.replace(/(?<!\x01)\b\d+(\.\d+)?([eE][+-]?\d+)?\b/g, function (m) { return mask(m, 'num'); });
    return restore(s, tokens);
  }

  function highlightCPP(code) {
    var tokens = []; var mask = maskFactory(tokens); var s = code;
    s = s.replace(/\/\*[\s\S]*?\*\//g, function (m) { return mask(m, 'com'); });
    s = s.replace(/\/\/[^\n]*/g, function (m) { return mask(m, 'com'); });
    s = s.replace(/&quot;[^&]*&quot;/g, function (m) { return mask(m, 'str'); });
    s = s.replace(/"[^"]*"/g, function (m) { return mask(m, 'str'); });
    s = s.replace(/#\s*include[^\n]*/g, function (m) { return mask(m, 'com'); });
    s = s.replace(API_NS, function (_, ns, fn) { return mask(ns, 'kw') + '::' + mask(fn, 'api'); });
    s = s.replace(API_POP, function (m) { return mask(m, 'api'); });
    s = s.replace(API_BARE, function (m) { return mask(m, 'api'); });
    s = s.replace(/\b(auto|const|return|if|else|for|while|void|struct|class|using|namespace|template|typename|public|private|override|final|static|inline|bool|true|false|nullptr|new|this|sizeof|continue|break|switch|case)\b/g, function (m) { return mask(m, 'kw'); });
    s = s.replace(/\b[A-Z][A-Za-z0-9_]+\b/g, function (m) { return mask(m, 'ty'); });
    s = s.replace(/(?<!\x01)\b\d+(\.\d+)?\b/g, function (m) { return mask(m, 'num'); });
    return restore(s, tokens);
  }

  function detectLang(text) {
    if (/#include|::|RewriterBase|LogicalResult|patterns\.add|applyPatterns|populate\w+Patterns|\brewriter\.|->\s*[A-Za-z]/.test(text)) return 'cpp';
    if (/\bfunc\.func\b|affine_map<|tensor<|memref<|iterator_types|%\w+\s*=|->\s*tensor|->\s*memref/.test(text)) return 'mlir';
    var pct = (text.match(/%[\w]/g) || []).length;
    var ns = (text.match(/::/g) || []).length;
    return pct >= ns ? 'mlir' : 'cpp';
  }

  function processBlock(el) {
    if (el.dataset.hl === '1') return;
    var explicit = (el.className.match(/language-(mlir|cpp)/) || [])[1];
    var lang = explicit || detectLang(el.textContent || '');
    el.innerHTML = (lang === 'cpp') ? highlightCPP(el.innerHTML) : highlightMLIR(el.innerHTML);
    el.classList.add('hl-' + lang);
    el.dataset.hl = '1';
  }

  function applyHighlight() {
    document.querySelectorAll('pre').forEach(function (pre) {
      var code = pre.querySelector('code');
      processBlock(code || pre);
    });
  }

  function injectCSS() {
    if (document.getElementById('code-hl-style')) return;
    var css = [
      '.tok-op{color:#ffb86c}', '.tok-kw{color:#ff79c6}', '.tok-ty{color:#8be9fd}',
      '.tok-val{color:#f1fa8c}', '.tok-num{color:#bd93f9}', '.tok-com{color:#7c8a94;font-style:italic}',
      '.tok-attr{color:#50fa7b}', '.tok-str{color:#f1fa8c}', '.tok-api{color:#ff5d6c;font-weight:700}',
      'code.api{background:#fbecd9;color:#8c3a00;border:1px solid #e6c79a;border-radius:4px;padding:.04em .36em;font-weight:700}',
      '.srcref{font-family:"JetBrains Mono","Fira Code",monospace;font-size:.76rem;color:#8a7d63;margin:-.55rem 0 1rem;display:flex;gap:.4rem;align-items:baseline;flex-wrap:wrap}',
      '.srcref::before{content:"\\1F4C4";font-style:normal}', '.srcref b{color:#6a5d44}',
      'table.apitbl td:first-child code{white-space:normal}'
    ].join('\n');
    var st = document.createElement('style');
    st.id = 'code-hl-style';
    st.textContent = css;
    document.head.appendChild(st);
  }

  function go() { injectCSS(); applyHighlight(); }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', go);
  else go();
})();
