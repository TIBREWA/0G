// Upload one episode file to 0G Storage, then record its root on 0G Chain.
// usage: node publish.mjs <file> <sha256> <decisions> <safeStops>
import { Indexer, ZgFile } from "@0gfoundation/0g-storage-ts-sdk";
import { ethers } from "ethers";

const RPC = process.env.ZG_RPC_URL ?? "https://evmrpc-testnet.0g.ai";
const INDEXER = process.env.ZG_INDEXER_URL ?? "https://indexer-storage-testnet-turbo.0g.ai";
const ABI = [
  "function record(bytes32 root, bytes32 logSha256, uint32 decisions, uint32 safeStops)",
  "function episodes(bytes32) view returns (bytes32, bytes32, uint64, uint32, uint32)",
];

const [file, sha256, decisions, safeStops] = process.argv.slice(2);
if (!process.env.ZG_PRIVATE_KEY || !process.env.ZG_REGISTRY_ADDRESS) {
  throw new Error("Set ZG_PRIVATE_KEY and ZG_REGISTRY_ADDRESS");
}
const provider = new ethers.JsonRpcProvider(RPC);
const signer = new ethers.Wallet(process.env.ZG_PRIVATE_KEY, provider);

const zgFile = await ZgFile.fromFilePath(file);
const [tree, treeErr] = await zgFile.merkleTree();
if (treeErr) throw treeErr;
const root = tree.rootHash();
const [upload, uploadErr] = await new Indexer(INDEXER).upload(zgFile, RPC, signer);
await zgFile.close();
if (uploadErr) throw uploadErr;

const registry = new ethers.Contract(process.env.ZG_REGISTRY_ADDRESS, ABI, signer);
const existing = await registry.episodes(root);
let recordTx = null;
if (existing[2] === 0n) {
  const tx = await registry.record(root, sha256, Number(decisions), Number(safeStops));
  await tx.wait();
  recordTx = tx.hash;
}
console.log(JSON.stringify({ root, uploadTx: upload.txHash ?? null, recordTx, registry: process.env.ZG_REGISTRY_ADDRESS }));
