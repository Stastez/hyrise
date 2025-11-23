#pragma once

#include <atomic>
#include <bit>
#include "types.hpp"

namespace hyrise {

/**
 * Frames are the metadata objects for each page. We use a 64-bit atomic integer to store the (latching) state, NUMA node, dirty flag, and the frame version.
 * All operations are atomic. The basic idea and most of the code is based on the SIGMOD'23 paper "Virtual-Memory Assisted Buffer Management" by Leis et al.
 *
 * The frame's upper 16 bits encode the (latching) state (see below). 1 bit is used for the dirty flag. 7 bits are used for the NUMA node. The lower 40 bits are used for the version.
 * The version is used to tract concurrent changes to the state of the frame. The version is incremented after exclusively unlocking the frame. It is not incremented when unlocking in shared mode.
 *
 *  +-----------+-------+-----------+----------------+
 *  | State     | Dirty | NUMA node | Version        |
 *  +-----------+-------+-----------+----------------+
 * 64          48      47          40                0
 *
 * The (latching) state is encoded as EVICTED (65535), LOCKED (65533), UNLOCKED (0), MARKED (65534) and LOCK_SHARED (1-65532). Initially, the frame is in state EVICTED.
 * Then, the frame gets LOCKED for reading from disk. After that, the frame is UNLOCKED and can be used. It can be LOCKED again for write access. For multiple current readers,
 * the state is incremented by 1 until MAX_LOCKED_SHARED is reached. When locking in shared mode, we also need to perform the same amount of unlocked to move the state to UNLOCKED again.
 * The state MARKED is used for Second-Chance marking a frame for eviction. Only frames that were previously MARKED after the UNLOCKED state are eligible for eviction. This approximates an LRU policy.
 * Later, the state is changed to EVICTED after the page has been written to disk. After unlocking a frame, the version counter is updated. The version is used to detect concurrent changes to the
 * state of the frame. The version is primarily used for the Second-Chance eviction mechanism. We use it to verify if an enqueued frame has been modified in the meantime and thereby is outdated. If so,
 * we know that a newer version of the frame has been enqueued or that the frame is currently locked. Thus, we can skip the frame for now. The version can also be used to implementet an optimistic latching
 * mechanism.
 *
 * The reference implementation can be found in https://github.com/viktorleis/vmcache/blob/master/vmcache.cpp.
 *
 *                                                            try_lock_exclusive
 *                                                   +----------------------------------------------------------------------+
 *                                                   v                                                                      |
 * +---------+  try_lock_exclusive                 +--------+  unlock_exclusive     +----------------+  try_mark          +---------------------+
 * | EVICTED | ----------------------------------> |        | --------------------> |                | -----------------> |       MARKED        |
 * +---------+                                     |        |                       |                |                    +---------------------+
 *   ^         unlock_exclusive_and_set_evicted    |        |  try_lock_exclusive   |                |                        try_lock_shared
 *   +-------------------------------------------- | LOCKED | <-------------------- |    UNLOCKED    |                      +-----------------+
 *                                                 |        |                       |                |                      v                 |
 *                                                 |        |                       |                |  try_lock_shared   +---------------------+
 *                                                 |        |                       |                | -----------------> |    LOCKED_SHARED    | -+
 *                                                 +--------+                       +----------------+                    +---------------------+  |
 *                                                                                    ^                                     ^ unlock_shared   |    |
 *                                                                                    | unlock_shared                       +-----------------+    |
 *                                                                                    |                                                            |
 *                                                                                    | (only if no other latches are present)                     |
 *                                                                                    +------------------------------------------------------------+
 */
class Frame {
 public:
  using StateVersionType = uint64_t;
  
  static constexpr StateVersionType UNLOCKED = 0;
  static constexpr StateVersionType LOCKED_SHARED = 0xFFFF - 3;  // 252 if 8 bits, 65532
  static constexpr StateVersionType LOCKED = 0xFFFF - 2;         // 253 if 8 bits, 65533
  static constexpr StateVersionType MARKED = 0xFFFF - 1;         // 254 if 8 bits, 65534
  static constexpr StateVersionType EVICTED = 0xFFFF;            // 255 if 8 bits, 65535 for 16 bits

  Frame();

  // Flags and metadata
  void set_node_id(const NodeID node_id);
  void set_dirty(const bool new_dirty);
  bool is_dirty() const;
  void reset_dirty();
  NodeID node_id() const;

  // State transitions
  void unlock_exclusive_and_set_evicted();

  bool try_mark(StateVersionType old_state_and_version);

  bool try_lock_shared(StateVersionType old_state_and_version);

  bool try_lock_exclusive(StateVersionType old_state_and_version);

  // Removes a shared locked and returns true if the frame is now unlocked
  bool unlock_shared();

  void unlock_exclusive();

  bool is_unlocked() const;

  // State and version helper
  StateVersionType state_and_version() const;
  static StateVersionType state(StateVersionType state_and_version);
  static StateVersionType version(StateVersionType state_and_version);
  static NodeID node_id(StateVersionType state_and_version);

  void debug_print();

 private:
  // clang-format off
  static constexpr uint64_t NODE_ID_MASK   = 0x00000F0000000000;
  static constexpr uint64_t DIRTY_MASK       = 0x0000F00000000000;
  static constexpr uint64_t STATE_MASK       = 0xFFFF000000000000;
  static constexpr uint64_t VERSION_MASK     = 0x000000FFFFFFFFFF;
  static_assert((NODE_ID_MASK ^ DIRTY_MASK ^ STATE_MASK ^ VERSION_MASK) == std::numeric_limits<StateVersionType>::max());
  // clang-format on

  static constexpr uint64_t NUM_BITS = sizeof(StateVersionType) * CHAR_BIT;
  static constexpr uint64_t NODE_ID_SHIFT = std::countr_zero(NODE_ID_MASK);
  static constexpr uint64_t DIRTY_SHIFT = std::countr_zero(DIRTY_MASK);
  static constexpr uint64_t STATE_SHIFT = std::countr_zero(STATE_MASK);

  StateVersionType update_state_with_same_version(StateVersionType old_version_and_state, StateVersionType new_state);
  StateVersionType update_state_with_increment_version(StateVersionType old_version_and_state,
                                                       StateVersionType new_state);

  std::atomic<StateVersionType> _state_and_version;
};

std::ostream& operator<<(std::ostream& os, const Frame& frame);
}  // namespace hyrise
