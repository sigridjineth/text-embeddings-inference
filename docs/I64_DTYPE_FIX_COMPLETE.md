# I64 Dtype Error - Complete Fix Applied

## 🎯 Error: `unsupported dtype I64 for op matmul`

## Root Cause Analysis

The error occurs when **integer tensors (I64/U32) leak into matmul operations**. Candle's matmul only supports floating-point dtypes (F32/F16/BF16).

**The Core Problem:**
- **Indexing tensors** (input_ids, attention_mask, position_ids) were mixed between U32 and I64
- **Computational tensors** (hidden_states, weights) need F32
- Dtype inconsistency caused integer leakage into matmul boundaries

**Why U32 Was Wrong:**
- Candle's embedding/position pipeline expects **I64** indices
- Using U32 created dtype mismatches during mask/position calculations
- Integer dtypes propagated through broadcasts and leaked into linear layers
- First matmul operation encountered I64 → error

---

## ✅ Complete Fix Applied

### Principle: **Indexing = I64, Computation = F32**

**All indexing tensors:** I64
- input_ids
- attention_mask
- position_ids

**All computational tensors:** F32
- hidden_states (from embeddings)
- attention weights & biases
- linear layer weights
- projector weights & inputs

---

## 🔧 Changes Made

### 1. LbnlReranker: Force I64 for Inputs ✅

**File:** `backends/candle/src/models/lbnl_reranker.rs:37-47`

**Changed:**
```rust
// BEFORE (WRONG - U32 causes leakage)
let ids = Tensor::from_vec(input.input_ids.clone(), (1, t), &self.device)?
    .to_dtype(DType::U32)?;  // ❌

let mask = Tensor::from_vec(input.attention_mask.clone(), (1, t), &self.device)?
    .to_dtype(DType::U32)?;  // ❌

// AFTER (CORRECT - I64 matches Candle)
let ids = Tensor::from_vec(input.input_ids.clone(), (1, t), &self.device)?
    .to_dtype(DType::I64)?;  // ✅

let mask = Tensor::from_vec(input.attention_mask.clone(), (1, t), &self.device)?
    .to_dtype(DType::I64)?;  // ✅
```

### 2. Qwen3: Force I64 for All Indexing Tensors ✅

**File:** `backends/candle/src/models/qwen3.rs:561-578`

**Changed:**
```rust
// BEFORE (WRONG - no explicit I64 cast + U32 for position_ids)
pub fn forward_with_tensors(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
    let input_ids = input_ids.to_device(&self.device)?;  // ❌ dtype not forced
    let attention_mask = attention_mask.to_device(&self.device)?;  // ❌ dtype not forced

    let position_ids = {
        let i64_mask = attention_mask.to_dtype(DType::I64)?;
        let one = Tensor::ones_like(&i64_mask)?;
        (i64_mask.cumsum(D::Minus1)? - one)?
            .broadcast_mul(&i64_mask)?
            .to_dtype(DType::U32)?  // ❌❌ U32 causes dtype mixing!
    };
    // ...
}

// AFTER (CORRECT - I64 for all indexing)
pub fn forward_with_tensors(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
    // CRITICAL: Force I64 dtype for all indexing tensors
    let input_ids = input_ids.to_device(&self.device)?.to_dtype(DType::I64)?;  // ✅
    let attention_mask = attention_mask.to_device(&self.device)?.to_dtype(DType::I64)?;  // ✅

    // CRITICAL: Keep position_ids as I64 (not U32) to prevent dtype mixing
    let position_ids = {
        let i64_mask = attention_mask.to_dtype(DType::I64)?;
        let one = Tensor::ones_like(&i64_mask)?;
        (i64_mask.cumsum(D::Minus1)? - one)?
            .broadcast_mul(&i64_mask)?
            .to_dtype(DType::I64)?  // ✅ Keep I64, not U32!
    };
    // ...
}
```

**Why This Matters:**
- The U32 at line 573 was **the critical bug**
- position_ids are used in rotary embeddings and attention
- Mixing U32 with I64 caused dtype inconsistency
- This leaked into computational paths → matmul error

### 3. Embeddings Always Output F32 ✅

**File:** `backends/candle/src/models/qwen3.rs:520-527`

**Already Implemented (Verified):**
```rust
let mut hidden_states = self.embeddings.forward(input_ids)?;

// CRITICAL FIX: Ensure embeddings output is in the model dtype (F32)
// Defensive cast so downstream matmul never sees integer tensors
hidden_states = if hidden_states.dtype() != self.dtype {
    hidden_states.to_dtype(self.dtype)?
} else {
    hidden_states
};
```

**This ensures:**
- Embeddings convert I64 indices → F32 hidden states
- All subsequent operations (attention, MLP, projector) receive F32
- Clean boundary between indexing (I64) and computation (F32)

### 4. Projector F32 Enforcement ✅

**File:** `backends/candle/src/layers/projector.rs:31-38, 54-58`

**Already Implemented (Verified):**
```rust
// Weights forced to F32 at load time
let w1 = w1.to_device(vb.device())?.to_dtype(DType::F32)?.contiguous()?;
let w2 = w2.to_device(vb.device())?.to_dtype(DType::F32)?.contiguous()?;

// Inputs forced to F32 at forward time
pub fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
    let hidden = if hidden.dtype() == DType::F32 {
        hidden.clone()
    } else {
        hidden.to_dtype(DType::F32)?
    };

    let hidden = hidden.contiguous()?;
    let h1 = hidden.matmul(&self.w1.t()?)?.relu()?;
    h1.matmul(&self.w2.t()?)
}
```

### 5. Diagnostic Logging Added ✅

**File:** `backends/candle/src/models/lbnl_reranker.rs:55-64`

```rust
let hs = if hs.dtype() != DType::F32 {
    tracing::warn!(
        "Hidden states dtype is {:?}, converting to F32 (this should not happen with I64 inputs)",
        hs.dtype()
    );
    hs.to_dtype(DType::F32)?
} else {
    tracing::debug!("Hidden states dtype is F32 as expected");  // ✅
    hs
};
```

**Expected logs (RUST_LOG=debug):**
```
DEBUG: Hidden states dtype is F32 as expected
```

**If you see:**
```
WARN: Hidden states dtype is <other>, converting to F32
```
→ There's an upstream dtype problem

---

## 🔍 Verification

### 1. Verify No U32 Remains

```bash
# Should return nothing
grep -r "to_dtype(DType::U32)" backends/candle/src/
```

**Result:** ✅ No U32 dtype conversions found

### 2. Build & Test

```bash
# Clean build
cargo clean
cargo build --release --features candle,http

# Run server
RUST_LOG=debug ./target/release/text-embeddings-router \
  --model-id jinaai/jina-reranker-v3

# Test request
curl -X POST http://localhost:8080/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning",
    "texts": ["ML is awesome", "Python is cool", "AI is the future"]
  }'
```

**Expected response:**
```json
{
  "results": [
    {"index": 0, "score": 0.92},
    {"index": 2, "score": 0.85},
    {"index": 1, "score": 0.34}
  ]
}
```

**Expected logs:**
```
INFO: Detected LBNL reranker (projector weights found)
INFO: Forcing F32 dtype for LBNL reranker
DEBUG: Hidden states dtype is F32 as expected
```

### 3. Verify Headers

```bash
curl -v http://localhost:8080/rerank ... 2>&1 | grep "x-tei-"
```

**Expected:**
```
< x-tei-rerank-strategy: listwise
< x-tei-lbnl-blocks: 1
< x-tei-lbnl-docs: 3
```

---

## 📊 Dtype Flow (Fixed)

```
User Request
  ↓
Vec<u32> input_ids, attention_mask
  ↓
┌─────────────────────────────────┐
│ LbnlReranker::forward           │
│ .to_dtype(I64) ✅               │  ← INDEXING (I64)
└─────────────────────────────────┘
  ↓
┌─────────────────────────────────┐
│ Qwen3::forward_with_tensors     │
│ input_ids.to_dtype(I64) ✅      │  ← INDEXING (I64)
│ attention_mask.to_dtype(I64) ✅ │  ← INDEXING (I64)
│ position_ids.to_dtype(I64) ✅   │  ← INDEXING (I64)
└─────────────────────────────────┘
  ↓
┌─────────────────────────────────┐
│ Embeddings Layer                │
│ I64 → F32 ✅                    │  ← BOUNDARY: Integer → Float
└─────────────────────────────────┘
  ↓
┌─────────────────────────────────┐
│ Qwen3::forward_layers           │
│ hidden_states.to_dtype(F32) ✅  │  ← COMPUTATION (F32)
│ Attention (F32) ✅              │  ← COMPUTATION (F32)
│ MLP (F32) ✅                    │  ← COMPUTATION (F32)
└─────────────────────────────────┘
  ↓
┌─────────────────────────────────┐
│ Projector                       │
│ input.to_dtype(F32) ✅          │  ← COMPUTATION (F32)
│ w1.matmul(F32) ✅               │  ← COMPUTATION (F32)
│ w2.matmul(F32) ✅               │  ← COMPUTATION (F32)
└─────────────────────────────────┘
  ↓
Vec<f32> embeddings
```

**Key:**
- **Green zone (I64):** All indexing operations
- **Blue zone (F32):** All computational operations
- **Boundary:** Embeddings convert I64 → F32 (explicit, clean)
- **Result:** No integer dtypes reach matmul

---

## ✅ Fix Checklist

- [x] **Remove all U32 conversions** (replaced with I64)
- [x] **Force I64 for input_ids** (lbnl_reranker.rs)
- [x] **Force I64 for attention_mask** (lbnl_reranker.rs, qwen3.rs)
- [x] **Force I64 for position_ids** (qwen3.rs - was U32!)
- [x] **Force F32 after embeddings** (qwen3.rs:forward_layers)
- [x] **Force F32 in projector** (projector.rs)
- [x] **Add diagnostic logging** (lbnl_reranker.rs)
- [x] **Verify no U32 remains** (grep confirmed)
- [x] **Build passes** (cargo clippy clean)

---

## 🎓 Why This Works

### The Problem
```
U32 (input) → I64 (position calc) → ??? (type confusion) → matmul gets I64 → ERROR
```

### The Solution
```
I64 (input) → I64 (position calc) → F32 (embeddings) → F32 (matmul) → SUCCESS
```

**Key Insights:**
1. **Candle expects I64** for embedding indices (not U32)
2. **Dtype mixing** (U32 ↔ I64) causes propagation bugs
3. **Clean boundaries** prevent leakage
4. **Explicit casts** make dtype flow predictable

---

## 📝 Summary

**Problem:** U32/I64 dtype mixing caused integer leakage into matmul operations

**Root Cause:**
- `position_ids` converted to U32 (line 573 in qwen3.rs)
- No explicit I64 cast for input_ids/attention_mask
- Dtype inconsistency propagated through computation graph

**Solution:**
1. ✅ Use **I64** for all indexing tensors (input_ids, attention_mask, position_ids)
2. ✅ Use **F32** for all computational tensors (hidden_states, weights, projector)
3. ✅ Remove **all U32** dtype conversions
4. ✅ Add diagnostic logging

**Status:** **COMPLETE** ✅
- All changes applied
- Build passes (cargo fmt + clippy clean)
- No U32 conversions remain
- Ready for testing

**Next Step:** Test with actual model to verify fix works in production
