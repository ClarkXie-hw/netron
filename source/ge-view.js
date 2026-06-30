
// Shared GE (Graph Engine) display layer for DaVinci OM and OM2 models.
// Converts ge.proto.ModelDef into Netron display objects (Model, Graph, Node, etc.).
// This module does NOT handle file I/O, zip parsing, or container format detection.

const ge = {};

ge.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading DaVinci OM model.';
    }
};

ge.Model = class {

    // container: { format, version, targets: [{ prefix, signature, model, weights }] }
    //   - format: display string, e.g. 'DaVinci OM', 'DaVinci OM SVP', 'DaVinci OM2'
    //   - version: version string (used for PICO; empty for others)
    //   - targets: one entry per ModelDef to render as modules in the Netron UI

    constructor(metadata, container) {
        this.format = container.format;
        this.version = container.version || '';
        this.modules = [];
        for (const target of container.targets) {
            const context = {
                metadata,
                signature: target.signature,
                weights: target.weights || null,
                prefix: target.prefix || ''
            };
            for (const graph of target.model.graph) {
                this.modules.push(new ge.Graph(context, graph));
            }
        }
    }
};

ge.Graph = class {

    constructor(context, graph) {
        // Graph naming by container signature:
        //   IMOD -> graph.name (e.g. "resnet50")
        //   PICO -> graph.id   (e.g. "0")
        //   OM2  -> prefix/graph.name (e.g. "model_0/resnet50_om2_dbg7")
        const name = graph.name || (graph.id === undefined ? '' : graph.id.toString());
        this.name = context.prefix ? `${context.prefix}/${name}` : name;
        this.nodes = [];
        this.inputs = [];
        this.outputs = [];
        const values = new Map();
        values.map = (name, type, tensor) => {
            if (values.has(name)) {
                if ((type && !type.equals(values.get(name).type)) ||
                    (tensor && tensor !== values.get(name).initializer)) {
                    throw new ge.Error(`Duplicate value '${name}'.`);
                }
            } else {
                values.set(name, new ge.Value(name, type || null, tensor || null));
            }
            return values.get(name);
        };
        // First pass: extract Const ops as initializers (tensors), skip them from node list
        const tensors = new Map();
        const ops = [];
        for (const op of graph.op) {
            if (op.type === 'Const' && op.attr && op.attr.value) {
                // Defensive: tensor definition may be incomplete in some OM2 protos
                const tensorDef = op.attr.value.t;
                if (!tensorDef || !tensorDef.desc) {
                    ops.push(op);
                    continue;
                }
                const desc = tensorDef.desc;
                let data = null;
                if (tensorDef.data && tensorDef.data.length !== 0) {
                    // Inline weight data embedded in the proto
                    data = tensorDef.data;
                } else if (context.weights === null) {
                    // No external weights available
                    data = null;
                } else {
                    // Compute weight size; guard against missing or non-numeric weight_size
                    const size = desc.weight_size && typeof desc.weight_size.toNumber === 'function'
                        ? desc.weight_size.toNumber() : 0;
                    if (size === 0) {
                        data = null;
                    } else if (desc.attr && desc.attr.merged_offset) {
                        // OM2-style merged offset into the weights buffer
                        const offset = desc.attr.merged_offset.i.toNumber();
                        data = context.weights.slice(offset, offset + size);
                    } else if (desc.data_offset && typeof desc.data_offset.toNumber === 'function') {
                        // Traditional OM data_offset into the weights buffer
                        const offset = desc.data_offset.toNumber();
                        data = context.weights.slice(offset, offset + size);
                    } else {
                        data = null;
                    }
                }
                const type = ge.Utility.tensorType(desc);
                const tensor = new ge.Tensor('Constant', type, data);
                tensors.set(op.name, tensor);
                continue;
            }
            ops.push(op);
        }
        // Second pass: build display nodes for non-Const ops
        for (const op of ops) {
            const node = new ge.Node(context, op, graph, values, tensors);
            this.nodes.push(node);
        }
    }
};

ge.Node = class {

    constructor(context, op, graph, values, tensors) {
        this.name = op.name || '';
        this.type = context.metadata.type(op.type) || { name: op.type };
        this.inputs = [];
        this.outputs = [];
        this.attributes = [];
        this.chain = [];
        this.controlDependencies = [];
        this.device = null;
        // Build input arguments from op.input list
        if (op.input) {
            let index = 0;
            for (let i = 0; i < op.input.length; i++) {
                const input = op.input[i];
                if (input === '') {
                    continue;
                }
                const name = this.type.inputs && i < this.type.inputs.length ? this.type.inputs[i].name : `input${index === 0 ? '' : index}`;
                index++;
                // Variadic tensor inputs consume remaining entries
                const end = this.type.inputs && i < this.type.inputs.length && this.type.inputs[i].type && this.type.inputs[i].type === 'Tensor[]' ? op.input.length : i + 1;
                const list = [];
                for (let j = i; j < end; j++) {
                    const input = op.input[j];
                    if (input === '') {
                        continue;
                    }
                    const index = input.lastIndexOf(':');
                    const identifier = input.substring(0, index);
                    const src_index = input.substring(index + 1);
                    // src_index === '-1' indicates a control dependency edge
                    if (src_index === '-1') {
                        this.controlDependencies.push(values.map(name));
                        continue;
                    }
                    const type = ge.Utility.tensorType(op.input_desc[j]);
                    const tensor = tensors.get(identifier);
                    const value = values.map(input, type, tensor);
                    list.push(value);
                }
                const argument = new ge.Argument(name, list);
                this.inputs.push(argument);
                i = end - 1;
            }
        }
        // Build output arguments from op.output_desc
        if (op.output_desc) {
            for (let i = 0; i < op.output_desc.length; i++) {
                const identifier = `${this.name}:${i}`;
                const type = ge.Utility.tensorType(op.output_desc[i]);
                const name = this.type.outputs && i < this.type.outputs.length ? this.type.outputs[i].name : `output${i === 0 ? '' : i}`;
                const value = values.map(identifier, type);
                const argument = new ge.Argument(name, [value]);
                this.outputs.push(argument);
            }
        }
        // Convert op.attr entries into display attributes
        for (const [name, obj] of Object.entries(op.attr || {})) {
            if (name === 'device') {
                this.device = obj;
                continue;
            }
            if (name === 'original_op_names') {
                continue;
            }
            // Fused ReLU activation shown as a chain node
            if (name === 'relu_flag' && obj.b) {
                const node = new ge.Node(context, { type: 'ReLU' }, graph, obj);
                this.chain.push(node);
                continue;
            }
            let value = obj;
            let type = null;
            switch (obj.value) {
                case 'i': {
                    value = obj.i;
                    type = 'int64';
                    break;
                }
                case 'f': {
                    value = obj.f;
                    type = 'float32';
                    break;
                }
                case 'b': {
                    value = obj.b;
                    type = 'boolean';
                    break;
                }
                case 'bt': {
                    // Byte tensor: interpret as float32 array
                    value = null;
                    if (obj.bt.length !== 0) {
                        type = 'tensor';
                        const shape = new ge.TensorShape([obj.bt.length / 4]);
                        value = new ge.Tensor('Constant', new ge.TensorType('float32', shape), obj.bt);
                    }
                    break;
                }
                case 'dt': {
                    type = 'DataType';
                    value = ge.Utility.dtype(Number(obj.dt));
                    break;
                }
                case 's': {
                    if (typeof obj.s === 'string') {
                        value = obj.s;
                    } else if (obj.s.every((c) => c >= 32 && c <= 128)) {
                        value = ge.Utility.decodeText(obj.s);
                    } else {
                        value = obj.s;
                    }
                    type = 'string';
                    break;
                }
                case 'g': {
                    type = 'graph';
                    value = new ge.Graph(context, obj.g);
                    break;
                }
                case 'func': {
                    type = 'function';
                    value = obj.func;
                    break;
                }
                case 'list': {
                    const list = obj.list;
                    value = [];
                    if (list.s && list.s.length > 0) {
                        // OM2 text proto decodes list.s as string arrays; legacy OM as byte arrays
                        value = list.s.map((v) => typeof v === 'string' ? v : ge.Utility.decodeText(v));
                        type = 'string[]';
                    } else if (list.b && list.b.length > 0) {
                        value = list.b;
                        type = 'boolean[]';
                    } else if (list.i && list.i.length > 0) {
                        value = list.i;
                        type = 'int64[]';
                    } else if (list.f && list.f.length > 0) {
                        value = list.f;
                        type = 'float32[]';
                    } else if (list.type && list.type.length > 0) {
                        type = 'type[]';
                        value = list.type.map((t) => {
                            try {
                                return ge.Utility.dtype(Number(t));
                            } catch {
                                return '?';
                            }
                        });
                    } else if (list.shape && list.shape.length > 0) {
                        type = 'shape[]';
                        value = list.shape.map((shape) => new ge.TensorShape(shape));
                    }
                    break;
                }
                case 'list_list_int': {
                    value = obj.list_list_int.list_list_i.map((list) => list.list_i);
                    break;
                }
                case 'list_list_float': {
                    value = obj.list_list_float.list_list_f.map((list) => list.list_f);
                    break;
                }
                case 'td': {
                    type = 'tensor type';
                    value = ge.Utility.tensorType(obj.td);
                    break;
                }
                case 't': {
                    type = 'tensor';
                    value = new ge.Tensor('Constant', ge.Utility.tensorType(obj.t.desc), obj.t.bytes);
                    break;
                }
                case undefined: {
                    value = null;
                    break;
                }
                default: {
                    throw new ge.Error(`Unsupported attribute type '${JSON.stringify(obj).substring(0, 32)}'.`);
                }
            }
            const attribute = new ge.Argument(name, value, type);
            this.attributes.push(attribute);
        }
    }
};

ge.Argument = class {

    constructor(name, value, type = null) {
        this.name = name;
        this.value = value;
        this.type = type;
    }
};

ge.Value = class {

    constructor(name, type, initializer = null) {
        if (typeof name !== 'string') {
            throw new ge.Error(`Invalid value identifier '${JSON.stringify(name)}'.`);
        }
        this.name = name;
        this.type = initializer ? initializer.type : type;
        this.initializer = initializer;
    }
};

ge.Tensor = class {

    constructor(category, type, value) {
        this.category = category;
        this.type = type;
        this.values = value;
    }
};

ge.TensorType = class {

    constructor(dataType, shape, denotation) {
        this.dataType = dataType;
        this.shape = shape;
        this.denotation = denotation;
    }

    equals(obj) {
        return obj && this.dataType === obj.dataType && this.shape && this.shape.equals(obj.shape);
    }

    toString() {
        return this.dataType + this.shape.toString();
    }
};

ge.TensorShape = class {

    constructor(dimensions) {
        this.dimensions = dimensions.map((dim) => typeof dim === 'bigint' ? dim.toNumber() : dim);
    }

    equals(obj) {
        if (obj && Array.isArray(obj.dimensions) && Array.isArray(this.dimensions)) {
            if (this.dimensions.length === obj.dimensions.length &&
                obj.dimensions.every((value, index) => this.dimensions[index] === value)) {
                return true;
            }
            if (obj.dimensions.every((dim) => Number.isInteger(dim)) && this.dimensions.every((dim) => Number.isInteger(dim))) {
                const a = obj.dimensions.reduce((a, b) => a * b, 1);
                const b = this.dimensions.reduce((a, b) => a * b, 1);
                return a === b;
            }
        }
        return false;
    }

    toString() {
        if (this.dimensions && Array.isArray(this.dimensions) && this.dimensions.length > 0) {
            return `[${this.dimensions.map((dim) => dim ? dim.toString() : '?').join(',')}]`;
        }
        return '';
    }
};

ge.Utility = class {

    // Map GE numeric dtype enum to human-readable string
    static dtype(value) {
        ge.Utility._types = ge.Utility._types || [
            'undefined', 'float32', 'float16', 'int8', 'uint8', 'int16', 'uint16', 'int32',
            'int64', 'uint32', 'uint64', 'boolean', 'float64', 'string', 'dual_sub_int8', 'dual_sub_uint8',
            'complex<float32>', 'complex<float64>', 'qint8', 'qint16', 'qint32', 'quint8', 'quint16', 'resource',
            'stringref', 'dual', 'variant', 'bfloat16', 'int4', 'uint1', 'int2', 'uint2'
        ];
        if (value >= ge.Utility._types.length) {
            throw new ge.Error(`Unsupported dtype '${value}'.`);
        }
        return ge.Utility._types[value];
    }

    // Build a TensorType from a TensorDesc (or similar shape descriptor)
    static tensorType(desc) {
        if (desc && desc.shape && Array.isArray(desc.shape.dim)) {
            const dataType = desc && desc.dtype ? ge.Utility.dtype(desc.dtype) : '?';
            const shape = new ge.TensorShape(desc.shape.dim);
            return new ge.TensorType(dataType, shape, desc.layout);
        }
        return null;
    }

    static decodeText(value) {
        ge.Utility._textDecoder = ge.Utility._textDecoder || new TextDecoder('utf-8');
        return ge.Utility._textDecoder.decode(value);
    }
};

export const Model = ge.Model;
export const Graph = ge.Graph;
export const Node = ge.Node;
export const Argument = ge.Argument;
export const Value = ge.Value;
export const Tensor = ge.Tensor;
export const TensorType = ge.TensorType;
export const TensorShape = ge.TensorShape;
export const Utility = ge.Utility;
const GeError = ge.Error;
export { GeError as Error };
