
// OM2 (DaVinci OM version 2) loader.
// OM2 packages are ZIP archives containing:
//   <prefix>/data/model_<n>/model_meta.json       - model metadata
//   <prefix>/data/model_<n>/debug/ge_proto_*.txt  - text protobuf (ge.proto.ModelDef)
//   <prefix>/data/constants/constant_<n>          - weight data (optional)
//
// The ZIP may have a top-level directory (e.g. "resnet50_om2_dbg7/data/..."),
// so all entry paths are normalized and matched with a flexible prefix.
//
// Unlike traditional OM (IMOD/PICO binary containers), OM2 uses text protobuf
// for graph structure and stores weights in separate constant files.

import * as ge from './ge-view.js';
import * as protobuf from './protobuf.js';

const om2 = {};

// Normalize backslashes to forward slashes for consistent matching
const normalize = (name) => name.replace(/\\/g, '/');

om2.ModelFactory = class {

    async match(context) {
        const entries = await context.peek('zip');
        if (!(entries instanceof Map) || entries.size === 0) {
            return null;
        }

        // Keep match() cheap. OM2 is a ZIP container and many unrelated formats
        // also use ZIP, so only scan entry names and return as soon as we see the
        // two structural markers required for visualization. Do not read weights
        // or decode ge_proto here; the full model list is built and validated in
        // open() after this loader has been selected.
        let hasMeta = false;
        let hasGeProto = false;
        for (const name of entries.keys()) {
            const path = normalize(name);
            if (!hasMeta && /(?:^|\/)data\/model_\d+\/model_meta\.json$/.test(path)) {
                hasMeta = true;
            }
            if (!hasGeProto && /(?:^|\/)data\/model_\d+\/debug\/ge_proto_.*\.txt$/.test(path)) {
                hasGeProto = true;
            }
            if (hasMeta && hasGeProto) {
                return context.set('om2', { entries });
            }
        }
        return null;
    }

    async open(context) {
        const target = context.value;
        const proto = await context.require('./om-proto');
        const metadata = await context.metadata('om-metadata.json');
        const targets = [];
        const models = om2.Utility.models(target.entries);

        for (const model of models) {
            // Optional weights. Graph structure remains viewable without this entry.
            const weightsName = om2.Utility.findEntry(target.entries, new RegExp(`(?:^|/)data/constants/constant_${model.index}$`));
            const weightsEntry = weightsName ? target.entries.get(weightsName) : null;
            const weights = weightsEntry ? weightsEntry.peek() : null;

            for (const name of model.proto) {
                const entry = target.entries.get(name);
                const buffer = entry.peek();

                const reader = protobuf.TextReader.open(buffer);
                if (!reader) {
                    throw new om2.Error(`File '${name}' is not a protobuf text file.`);
                }

                let modelDef = null;
                try {
                    modelDef = proto.ge.proto.ModelDef.decodeText(reader);
                } catch (error) {
                    const message = error && error.message ? error.message : error.toString();
                    throw new om2.Error(`File '${name}' is not ge.proto.ModelDef (${message.replace(/\.$/, '')}).`);
                }

                // When a model has multiple ge_proto files, disambiguate graph names
                // by including the proto filename in the prefix
                const protoBase = model.proto.length > 1
                    ? name.split('/').pop().replace(/^ge_proto_/, '').replace(/\.txt$/, '')
                    : '';
                const prefix = protoBase
                    ? `model_${model.index}/${protoBase}`
                    : `model_${model.index}`;

                targets.push({
                    prefix,
                    signature: 'OM2',
                    model: modelDef,
                    weights
                });
            }
        }

        return new ge.Model(metadata, {
            format: 'DaVinci OM2',
            version: '',
            targets
        });
    }
};

om2.Error = class extends Error {
    constructor(message) {
        super(message);
        this.name = 'Error loading DaVinci OM2 model.';
    }
};

om2.Utility = class {

    static models(entries) {
        // Build the complete OM2 model table once the loader has been selected.
        // At this point failures should be reported as OM2 errors instead of
        // silently falling through to other ZIP-based formats.
        const models = new Map();
        for (const name of entries.keys()) {
            const path = normalize(name);
            const match = path.match(/(?:^|\/)data\/model_(\d+)\/model_meta\.json$/);
            if (match) {
                const index = Number(match[1]);
                models.set(index, {
                    index,
                    meta: name,
                    proto: []
                });
            }
        }
        if (models.size === 0) {
            throw new om2.Error('File does not contain OM2 model metadata.');
        }
        for (const name of entries.keys()) {
            const path = normalize(name);
            const match = path.match(/(?:^|\/)data\/model_(\d+)\/debug\/ge_proto_.*\.txt$/);
            if (match) {
                const index = Number(match[1]);
                if (models.has(index)) {
                    models.get(index).proto.push(name);
                }
            }
        }
        const result = Array.from(models.values()).sort((a, b) => a.index - b.index);
        for (const model of result) {
            if (model.proto.length === 0) {
                throw new om2.Error(`OM2 model_${model.index} does not contain a GE graph proto.`);
            }
            model.proto.sort();
        }
        return result;
    }

    static findEntry(entries, pattern) {
        for (const name of entries.keys()) {
            if (pattern.test(normalize(name))) {
                return name;
            }
        }
        return null;
    }
};

export const ModelFactory = om2.ModelFactory;
