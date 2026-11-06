// SPDX-License-Identifier: MIT
pragma solidity 0.8.19;

import {EpisodeRegistry} from "../src/EpisodeRegistry.sol";

interface Vm {
    function prank(address sender) external;
    function warp(uint256 timestamp) external;
    function expectRevert(bytes4 selector) external;
}

contract EpisodeRegistryTest {
    Vm private constant vm = Vm(address(uint160(uint256(keccak256("hevm cheat code")))));
    address private constant RECORDER = address(0xA11CE);
    EpisodeRegistry private registry;

    function setUp() public {
        vm.warp(1000);
        registry = new EpisodeRegistry(RECORDER);
    }

    function _record(bytes32 root) private {
        vm.prank(RECORDER);
        registry.record(root, keccak256(abi.encode(root)), 4, 1);
    }

    function test_RecordStoresEpisode() public {
        _record("a");
        (bytes32 sha, bytes32 prev, uint64 at, uint32 decisions, uint32 stops) = registry.episodes("a");
        require(sha == keccak256(abi.encode(bytes32("a"))), "sha");
        require(prev == bytes32(0) && at == 1000 && decisions == 4 && stops == 1, "fields");
        require(registry.lastRoot() == "a" && registry.episodeCount() == 1, "index");
    }

    function test_EpisodesChainTogether() public {
        _record("a");
        _record("b");
        (, bytes32 prev,,,) = registry.episodes("b");
        require(prev == "a", "chain");
    }

    function test_OnlyRecorderCanWrite() public {
        vm.expectRevert(EpisodeRegistry.NotRecorder.selector);
        registry.record("a", "x", 1, 0);
    }

    function test_DuplicateRootReverts() public {
        _record("a");
        vm.prank(RECORDER);
        vm.expectRevert(EpisodeRegistry.AlreadyRecorded.selector);
        registry.record("a", "x", 1, 0);
    }

    function test_EmptyRootReverts() public {
        vm.prank(RECORDER);
        vm.expectRevert(EpisodeRegistry.EmptyRoot.selector);
        registry.record(bytes32(0), "x", 1, 0);
    }
}
