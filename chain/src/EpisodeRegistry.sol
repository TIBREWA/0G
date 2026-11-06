// SPDX-License-Identifier: MIT
pragma solidity 0.8.19;

/// @notice Tamper-evident index of rover episodes stored on 0G Storage.
/// Each entry links to the previous one, so a run cannot be dropped or reordered later.
contract EpisodeRegistry {
    struct Episode {
        bytes32 logSha256;
        bytes32 prevRoot;
        uint64 recordedAt;
        uint32 decisions;
        uint32 safeStops;
    }

    address public immutable recorder;
    bytes32 public lastRoot;
    uint256 public episodeCount;
    mapping(bytes32 => Episode) public episodes;

    event EpisodeRecorded(bytes32 indexed root, bytes32 indexed prevRoot, bytes32 logSha256, uint32 decisions, uint32 safeStops);

    error NotRecorder();
    error EmptyRoot();
    error AlreadyRecorded();

    constructor(address recorder_) {
        recorder = recorder_;
    }

    function record(bytes32 root, bytes32 logSha256, uint32 decisions, uint32 safeStops) external {
        if (msg.sender != recorder) revert NotRecorder();
        if (root == bytes32(0)) revert EmptyRoot();
        if (episodes[root].recordedAt != 0) revert AlreadyRecorded();

        bytes32 prevRoot = lastRoot;
        episodes[root] = Episode(logSha256, prevRoot, uint64(block.timestamp), decisions, safeStops);
        lastRoot = root;
        episodeCount += 1;
        emit EpisodeRecorded(root, prevRoot, logSha256, decisions, safeStops);
    }
}
