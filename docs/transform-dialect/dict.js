/* mlir-dict — MLIR/C++ 신택스 하이라이팅 + before/after 자동 라인 diff 강조.
   자체완결(외부 의존 없음). DOMContentLoaded에서 자동 실행.
   사용 규약:
     <pre><code class="mlir">...</code></pre>   // MLIR 코드 (자동 하이라이트)
     <pre><code class="cpp">...</code></pre>    // C++ 호출 줄
     <div class="ba">                              // before/after 묶음 (자동 diff)
       <div class="ba-grid">
         <div class="ba-col before"><div class="col-lbl">before</div><pre><code class="mlir">…</code></pre></div>
         <div class="ba-col after"> <div class="col-lbl">after</div> <pre><code class="mlir">…</code></pre></div>
       </div></div>
   코드 텍스트는 평문(HTML escape는 작성자가; 이 스크립트가 textContent를 다시 안전 변환). */
(function(){
  "use strict";
  function esc(s){return s.replace(/[&<>]/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;'}[c];});}

  // ── MLIR 토크나이저 ──────────────────────────────────────────────
  var MLIR_TYPES = /^(tensor|memref|vector|index|none|complex|i1|i2|i4|i8|i16|i32|i64|i128|si8|si16|si32|si64|ui8|ui16|ui32|ui64|f8|f16|f32|f64|bf16|f80|f128)$/;
  var MLIR_KW = /^(func|func\.func|return|module|ins|outs|iterator_types|indexing_maps|parallel|reduction|window|dense|true|false|loc|attributes|to|step|affine_map|affine_set|unit)$/;
  // 우선순위 순 alternation. 그룹: 1 comment, 2 string, 3 affine_map/set, 4 ssa/block/attr,
  // 5.6 dialect.op, 7 number, 8 word
  var MLIR_RE = /(\/\/[^\n]*)|("(?:[^"\\]|\\.)*")|\b(affine_map|affine_set)\b|(%[A-Za-z0-9_]+(?::[0-9]+)?|\^bb[0-9]+|#[A-Za-z_][A-Za-z0-9_]*)|\b([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]+)|(-?\b\d+\.?\d*(?:e[+-]?\d+)?\b)|([A-Za-z_][A-Za-z0-9_]*)/g;

  function hlMLIR(line){
    var out="", last=0, m;
    MLIR_RE.lastIndex=0;
    while((m=MLIR_RE.exec(line))){
      out += esc(line.slice(last,m.index));
      if(m[1]) out += '<span class="t-comment">'+esc(m[1])+'</span>';
      else if(m[2]) out += '<span class="t-str">'+esc(m[2])+'</span>';
      else if(m[3]) out += '<span class="t-map">'+esc(m[3])+'</span>';
      else if(m[4]) out += '<span class="t-ssa">'+esc(m[4])+'</span>';
      else if(m[5]!==undefined && m[6]!==undefined)
        out += '<span class="t-dialect">'+esc(m[5])+'</span>.<span class="t-op">'+esc(m[6])+'</span>';
      else if(m[7]) out += '<span class="t-num">'+esc(m[7])+'</span>';
      else if(m[8]){
        var w=m[8];
        if(MLIR_TYPES.test(w)) out += '<span class="t-type">'+esc(w)+'</span>';
        else if(MLIR_KW.test(w)) out += '<span class="t-kw">'+esc(w)+'</span>';
        else out += esc(w);
      }
      last = MLIR_RE.lastIndex;
    }
    out += esc(line.slice(last));
    return out;
  }

  // ── C++ 토크나이저 (호출 줄용, 경량) ─────────────────────────────
  var CPP_KW = /^(void|auto|return|const|constexpr|struct|class|public|private|protected|override|final|namespace|using|for|if|else|while|true|false|nullptr|static|inline|template|typename|bool|int|unsigned|size_t|std|mlir|linalg|affine|tensor|memref)$/;
  var CPP_RE = /(\/\/[^\n]*|\/\*[\s\S]*?\*\/)|("(?:[^"\\]|\\.)*")|(-?\b\d+\.?\d*\b)|([A-Za-z_][A-Za-z0-9_]*)/g;
  function hlCPP(line){
    var out="", last=0, m;
    CPP_RE.lastIndex=0;
    while((m=CPP_RE.exec(line))){
      out += esc(line.slice(last,m.index));
      if(m[1]) out += '<span class="t-comment">'+esc(m[1])+'</span>';
      else if(m[2]) out += '<span class="t-str">'+esc(m[2])+'</span>';
      else if(m[3]) out += '<span class="t-num">'+esc(m[3])+'</span>';
      else if(m[4]) out += CPP_KW.test(m[4]) ? '<span class="t-kw">'+esc(m[4])+'</span>' : esc(m[4]);
      last = CPP_RE.lastIndex;
    }
    out += esc(line.slice(last));
    return out;
  }

  function hl(line, lang){ return lang==="cpp" ? hlCPP(line) : hlMLIR(line); }

  // ── LCS 라인 diff: before/after 각 라인을 common/del/add로 분류 ───
  function lineDiff(A, B){
    var n=A.length, m=B.length, dp=[];
    for(var i=0;i<=n;i++){ dp.push(new Array(m+1).fill(0)); }
    for(var i=n-1;i>=0;i--)
      for(var j=m-1;j>=0;j--)
        dp[i][j] = (A[i].trim()===B[j].trim()) ? dp[i+1][j+1]+1 : Math.max(dp[i+1][j], dp[i][j+1]);
    var aTag=new Array(n).fill("del"), bTag=new Array(m).fill("add");
    var i=0,j=0;
    while(i<n && j<m){
      if(A[i].trim()===B[j].trim()){ aTag[i]="same"; bTag[j]="same"; i++; j++; }
      else if(dp[i+1][j] >= dp[i][j+1]){ aTag[i]="del"; i++; }
      else { bTag[j]="add"; j++; }
    }
    return {aTag:aTag, bTag:bTag};
  }

  function renderLines(codeEl, lang, tags){
    var lines = codeEl.textContent.replace(/\n$/,"").split("\n");
    var html = lines.map(function(ln, idx){
      var cls = tags ? ({same:"", del:"d-del", add:"d-add"}[tags[idx]]||"") : "";
      var inner = hl(ln, lang) || "​";
      return '<span class="ln '+cls+'">'+inner+'</span>';
    }).join("\n");
    codeEl.innerHTML = html;
  }

  function highlightStandalone(codeEl){
    var lang = codeEl.classList.contains("cpp") ? "cpp" : "mlir";
    // 라인 wrapping 없이 인라인 하이라이트 (pre가 \n 보존)
    var lines = codeEl.textContent.replace(/\n$/,"").split("\n");
    codeEl.innerHTML = lines.map(function(ln){return hl(ln, lang);}).join("\n");
    codeEl.dataset.hl = "1";
  }

  function setupBA(ba){
    var grid = ba.querySelector(".ba-grid");
    if(!grid) return;
    var beforeCode = ba.querySelector(".ba-col.before code");
    var afterCode  = ba.querySelector(".ba-col.after code");
    if(beforeCode && afterCode){
      var lang = afterCode.classList.contains("cpp") ? "cpp" : "mlir";
      var A = beforeCode.textContent.replace(/\n$/,"").split("\n");
      var B = afterCode.textContent.replace(/\n$/,"").split("\n");
      var d = lineDiff(A, B);
      renderLines(beforeCode, lang, d.aTag);
      renderLines(afterCode,  lang, d.bTag);
    } else {
      // single-block ba: 그냥 하이라이트
      ba.querySelectorAll("code").forEach(function(c){ if(!c.dataset.hl) highlightStandalone(c); });
    }
    ba.classList.add("diff-on");
    // 토글 버튼 (head에 있으면 사용, 없으면 생성)
    var btn = ba.querySelector(".ba-toggle");
    if(!btn){
      var head = ba.querySelector(".ba-head");
      btn = document.createElement("button");
      btn.className = "ba-toggle"; btn.type="button"; btn.textContent="변경 강조";
      if(head) head.appendChild(btn);
      else { var h=document.createElement("div"); h.className="ba-head"; h.appendChild(btn); ba.insertBefore(h, ba.firstChild); }
    }
    btn.setAttribute("aria-pressed","true");
    btn.addEventListener("click", function(){
      var on = ba.classList.toggle("diff-on");
      btn.setAttribute("aria-pressed", String(on));
    });
  }

  // ── 트리 전체 펼치기/접기 ────────────────────────────────────────
  function setupTrees(){
    document.querySelectorAll(".tree-tools").forEach(function(tools){
      var tree = tools.nextElementSibling;
      while(tree && !tree.classList.contains("tree")) tree = tree.nextElementSibling;
      if(!tree) return;
      tools.querySelectorAll("button[data-act]").forEach(function(b){
        b.addEventListener("click", function(){
          var open = b.dataset.act === "expand";
          tree.querySelectorAll("details").forEach(function(d){ d.open = open; });
        });
      });
    });
  }

  // ── 상세(.entry) → 간략 목록(가장 가까운 앞 h2[id])으로 복귀 링크 주입 ──
  function addBackToList(){
    document.querySelectorAll(".entry[id]").forEach(function(entry){
      var head = entry.querySelector(".entry-head");
      if(!head || head.querySelector(".tolist")) return;
      // 통합 목록(#list)이 있으면 그쪽으로, 없으면 가장 가까운 앞 h2[id]
      var tgt = null;
      if(document.getElementById("list")) tgt = "list";
      else { var n = entry.previousElementSibling; while(n){ if(n.tagName === "H2" && n.id){ tgt = n.id; break; } n = n.previousElementSibling; } }
      var a = document.createElement("a");
      a.className = "tolist";
      a.href = tgt ? ("#" + tgt) : "#top";
      a.textContent = "↑ 목록";
      a.title = "목록으로";
      var anchor = head.querySelector(".anchor");
      if(anchor) head.insertBefore(a, anchor); else head.appendChild(a);
    });
  }

  function run(){
    // 0) 상세 → 목록 복귀 링크
    addBackToList();
    // 1) ba 묶음 먼저 (diff + 하이라이트)
    document.querySelectorAll(".ba").forEach(setupBA);
    // 2) 나머지 standalone 코드 블록
    document.querySelectorAll("pre code.mlir, pre code.cpp").forEach(function(c){
      if(!c.dataset.hl && !c.closest(".ba")) highlightStandalone(c);
    });
    // 3) 트리 도구
    setupTrees();
  }
  if(document.readyState==="loading") document.addEventListener("DOMContentLoaded", run);
  else run();
})();
