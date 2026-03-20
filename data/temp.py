
# REPOSITORIOS DE GITHUB
REPOS_GITHUB = [

    ("huggingface", "transformers", "main", [
        "src/transformers/modeling_utils.py",
        "src/transformers/trainer.py",
        "src/transformers/tokenization_utils.py",
        "src/transformers/configuration_utils.py",
        "src/transformers/models/llama/modeling_llama.py",
        "src/transformers/models/gpt2/modeling_gpt2.py",
        "src/transformers/models/bert/modeling_bert.py",
        "src/transformers/generation/utils.py",
        "src/transformers/pipelines/base.py",
        "examples/pytorch/language-modeling/run_clm.py",
        "examples/pytorch/text-classification/run_glue.py",
    ]),
    ("pytorch", "pytorch", "main", [
        "torch/nn/modules/transformer.py",
        "torch/nn/modules/attention.py",
        "torch/optim/adam.py",
        "torch/optim/sgd.py",
        "torch/optim/lr_scheduler.py",
        "torch/utils/data/dataset.py",
        "torch/utils/data/dataloader.py",
        "torch/autograd/__init__.py",
        "torch/jit/_recursive.py",
        "torch/distributed/distributed_c10d.py",
    ]),
    ("karpathy", "nanoGPT", "master", [
        "model.py",
        "train.py",
        "data/shakespeare_char/prepare.py",
        "config/train_gpt2.py",
        "config/train_shakespeare_char.py",
    ]),
    ("karpathy", "llm.c", "master", [
        "train_gpt2.c",
        "train_gpt2.cu",
        "llm.h",
        "dev/data/tinyshakespeare.py",
    ]),
    ("karpathy", "llama2.c", "master", [
        "model.c",
        "run.c",
        "export.py",
    ]),
    ("karpathy", "minGPT", "master", [
        "mingpt/model.py",
        "mingpt/trainer.py",
        "mingpt/utils.py",
        "mingpt/bpe.py",
    ]),
    ("ray-project", "ray", "master", [
        "python/ray/worker.py",
        "python/ray/node.py",
        "python/ray/autoscaler/node_provider.py",
        "python/ray/train/trainer.py",
        "python/ray/serve/api.py",
        "python/ray/dashboard/dashboard.py",
    ]),
    ("langchain-ai", "langchain", "main", [
        "libs/langchain/langchain/chains/base.py",
        "libs/langchain/langchain/llms/base.py",
        "libs/langchain/langchain/vectorstores/base.py",
        "libs/langchain/langchain/agents/agent.py",
        "libs/langchain/langchain/prompts/prompt.py",
        "libs/langchain/langchain/memory/buffer.py",
    ]),
    ("run-llama", "llama_index", "main", [
        "llama-index-core/llama_index/core/indices/base.py",
        "llama-index-core/llama_index/core/node_parser/interface.py",
        "llama-index-core/llama_index/core/query_engine/retriever_query_engine.py",
        "llama-index-integrations/llms/llama-index-llms-openai/llama_index/llms/openai/base.py",
        "poetry.lock",
    ]),
    ("vllm-project", "vllm", "main", [
        "vllm/engine/llm_engine.py",
        "vllm/model_executor/models/llama.py",
        "vllm/sampling_params.py",
        "vllm/entrypoints/api_server.py",
    ]),
    ("bentoml", "BentoML", "main", [
        "src/bentoml/_internal/service/service.py",
        "src/bentoml/_internal/models/model.py",
    ]),
    ("Significant-Gravitas", "AutoGPT", "master", [
        "autogpt/agent/agent.py",
        "autogpt/commands/command.py",
    ]),
    ("google", "jax", "main", [
        "jax/_src/core.py",
        "jax/_src/numpy/lax_numpy.py",
    ]),
    ("fastai", "fastai", "master", [
        "fastai/learner.py",
        "fastai/optimizer.py",
        "fastai/callback/core.py",
        "fastai/data/core.py",
        "nbs/02_data.load.ipynb",
    ]),
    ("scikit-learn", "scikit-learn", "main", [
        "sklearn/linear_model/_base.py",
        "sklearn/ensemble/_forest.py",
        "sklearn/neural_network/_multilayer_perceptron.py",
        "sklearn/preprocessing/_data.py",
        "sklearn/model_selection/_split.py",
    ]),
    ("tiangolo", "fastapi", "master", [
        "fastapi/applications.py",
        "fastapi/routing.py",
        "fastapi/dependencies/utils.py",
        "fastapi/security/oauth2.py",
        "fastapi/params.py",
        "fastapi/responses.py",
        "docs_src/bigger_applications/app/main.py",
    ]),
    ("pallets", "flask", "main", [
        "src/flask/app.py",
        "src/flask/routing.py",
        "src/flask/blueprints.py",
        "src/flask/testing.py",
        "src/flask/sessions.py",
        "src/flask/ctx.py",
    ]),
    ("sqlalchemy", "sqlalchemy", "main", [
        "lib/sqlalchemy/orm/session.py",
        "lib/sqlalchemy/orm/query.py",
        "lib/sqlalchemy/engine/base.py",
        "lib/sqlalchemy/sql/selectable.py",
    ]),
    ("psf", "requests", "main", [
        "requests/api.py",
        "requests/sessions.py",
        "requests/models.py",
        "requests/auth.py",
        "requests/adapters.py",
    ]),
    ("python", "cpython", "main", [
        "Lib/asyncio/tasks.py",
        "Lib/asyncio/events.py",
        "Lib/asyncio/base_events.py",
        "Lib/collections/__init__.py",
        "Lib/functools.py",
        "Lib/itertools.py",
        "Lib/pathlib.py",
        "Lib/json/__init__.py",
        "Lib/logging/__init__.py",
        "Lib/threading.py",
        "Lib/multiprocessing/pool.py",
        "Lib/typing.py",
    ]),
    ("pydantic", "pydantic", "main", [
        "pydantic/main.py",
        "pydantic/fields.py",
        "pydantic/validators.py",
        "pydantic/_internal/_model_construction.py",
    ]),

    ("microsoft", "TypeScript", "main", [
        "src/compiler/checker.ts",
        "src/compiler/parser.ts",
        "src/compiler/emitter.ts",
        "src/compiler/types.ts",
        "src/compiler/utilities.ts",
    ]),
    ("vercel", "next.js", "canary", [
        "packages/next/src/server/app-router.ts",
        "packages/next/src/client/router.ts",
        "packages/next/src/lib/router/utils/route-matcher.ts",
        "packages/next/src/server/base-server.ts",
        "packages/next/src/build/index.ts",
        "packages/next-swc/crates/core/src/lib.rs",
    ]),
    ("vitejs", "vite", "main", [
        "packages/vite/src/node/server/index.ts",
        "packages/vite/src/node/build.ts",
        "packages/vite/src/node/config.ts",
        "packages/vite/src/node/plugins/resolve.ts",
    ]),
    ("angular", "angular", "main", [
        "packages/core/src/application/application_ref.ts",
        "packages/core/src/change_detection/change_detection_util.ts",
        "packages/router/src/router.ts",
        "packages/common/src/pipes/date_pipe.ts",
    ]),
    ("facebook", "react", "main", [
        "packages/react/src/React.js",
        "packages/react-reconciler/src/ReactFiber.js",
        "packages/react-dom/src/client/ReactDOMRoot.js",
    ]),
    ("expressjs", "express", "master", [
        "lib/application.js",
        "lib/router/index.js",
        "lib/router/route.js",
        "lib/middleware/json.js",
        "lib/response.js",
    ]),
    ("prisma", "prisma", "main", [
        "packages/client/src/runtime/core/engines/common/Engine.ts",
        "packages/client/src/runtime/getPrismaClient.ts",
        "packages/internals/src/engine-commands/getSchema.ts",
    ]),
    ("remix-run", "remix", "main", [
        "packages/remix-server-runtime/index.ts",
        "packages/remix-dev/compiler/index.ts",
    ]),
    ("sveltejs", "svelte", "main", [
        "packages/svelte/src/compiler/index.js",
    ]),
    ("withastro", "astro", "main", [
        "packages/astro/src/core/render/index.ts",
    ]),
    ("solidjs", "solid", "main", [
        "packages/solid/src/reactive/signal.ts",
    ]),
    ("fastify", "fastify", "main", [
        "lib/reply.js",
    ]),
    ("rust-lang", "rust", "master", [
        "library/std/src/collections/hash/map.rs",
        "library/std/src/io/mod.rs",
        "library/std/src/fs.rs",
        "library/core/src/iter/mod.rs",
        "library/core/src/option.rs",
        "library/core/src/result.rs",
        "compiler/rustc_middle/src/ty/mod.rs",
    ]),
    ("tokio-rs", "tokio", "master", [
        "tokio/src/runtime/mod.rs",
        "tokio/src/runtime/scheduler/multi_thread/mod.rs",
        "tokio/src/task/mod.rs",
        "tokio/src/net/tcp/stream.rs",
        "tokio/src/sync/mpsc/mod.rs",
    ]),
    ("serde-rs", "serde", "master", [
        "serde/src/lib.rs",
        "serde/src/ser/mod.rs",
        "serde/src/de/mod.rs",
        "serde_derive/src/ser.rs",
        "serde_derive/src/de.rs",
    ]),
    ("actix", "actix-web", "master", [
        "actix-web/src/app.rs",
        "actix-web/src/server.rs",
        "actix-web/src/request.rs",
        "actix-web/src/response/response.rs",
        "actix-web/src/middleware/logger.rs",
    ]),
    ("tauri-apps", "tauri", "dev", [
        "core/tauri/src/lib.rs",
    ]),
    ("BurntSushi", "ripgrep", "master", [
        "crates/core/main.rs",
    ]),
    ("nushell", "nushell", "main", [
        "crates/nu-cli/src/main.rs",
    ]),

    ("gin-gonic", "gin", "master", [
        "gin.go",
        "context.go",
        "routergroup.go",
        "tree.go",
        "middleware.go",
        "render/render.go",
    ]),
    ("ollama", "ollama", "main", [
        "server/routes.go",
        "app/lifecycle/lifecycle.go",
        "api/types.go",
        "llama/llama.go",
    ]),
    ("gohugoio", "hugo", "master", [
        "commands/hugo.go",
        "tpl/tpl.go",
        "hugolib/site.go",
    ]),
    ("prometheus", "prometheus", "main", [
        "main.go",
        "promql/engine.go",
        "storage/remote/remote.go",
    ]),
    ("grafana", "grafana", "main", [
        "pkg/api/api.go",
        "pkg/services/sqlstore/sqlstore.go",
        "pkg/middleware/middleware.go",
    ]),

    ("moby", "moby", "master", [
        "daemon/daemon.go",
        "client/client.go",
        "api/server/server.go",
        "image/image.go",
    ]),
    ("kubernetes", "kubernetes", "master", [
        "pkg/scheduler/scheduler.go",
        "pkg/kubelet/kubelet.go",
        "staging/src/k8s.io/api/core/v1/types.go",
        "pkg/apis/core/v1/validation/validation.go",
        "cmd/kube-apiserver/app/server.go",
    ]),
    ("hashicorp", "vault", "main", [
        "vault/logical_system.go",
        "vault/core.go",
        "sdk/helper/jsonutil/jsonutil.go",
    ]),
    ("hashicorp", "terraform", "main", [
        "internal/command/apply.go",
        "internal/terraform/context.py",
        "internal/backend/local/backend.go",
    ]),
    ("redis", "redis", "unstable", [
        "src/server.c",
        "src/object.c",
        "src/networking.c",
        "src/evict.c",
    ]),
    ("supabase", "supabase", "master", [
        "apps/www/lib/supabase.ts",
        "packages/common/gotrue.ts",
        "docker/docker-compose.yml",
    ]),
    ("clickhouse", "clickhouse", "master", [
        "src/Interpreters/InterpreterSelectQuery.cpp",
    ]),
    ("duckdb", "duckdb", "master", [
        "src/main/connection.cpp",
    ]),
    ("cockroachdb", "cockroach", "master", [
        "pkg/sql/exec/engine.go",
    ]),
    ("milvus-io", "milvus", "master", [
        "internal/querynode/segments.go",
    ]),

    ("microsoft", "vscode", "main", [
        "src/vs/editor/common/model/textModel.ts",
        "src/vs/platform/extensions/common/extensions.ts",
        "src/vs/workbench/api/common/extHost.api.impl.ts",
        "src/vs/editor/browser/view/viewPart.ts",
        "src/vs/base/common/lifecycle.ts",
        "src/vs/workbench/browser/layout.ts",
    ]),

    ("cypress-io", "cypress", "develop", [
        "packages/driver/src/cypress/cy.ts",
        "packages/server/lib/project.js",
        "packages/runner/src/main.tsx",
    ]),

    ("torvalds", "linux", "master", [
        "kernel/sched/core.c",
        "kernel/fork.c",
        "kernel/exit.c",
        "kernel/exec.c",
        "mm/memory.c",
        "fs/read_write.c",
        "net/ipv4/tcp.c",
    ]),

    ("spring-projects", "spring-framework", "main", [
        "spring-webmvc/src/main/java/org/springframework/web/servlet/DispatcherServlet.java",
        "spring-core/src/main/java/org/springframework/core/env/Environment.java",
    ]),
    ("elastic", "elasticsearch", "main", [
        "server/src/main/java/org/elasticsearch/node/Node.java",
        "server/src/main/java/org/elasticsearch/cluster/ClusterState.java",
    ]),

    ("bitcoin", "bitcoin", "master", [
        "src/validation.cpp",
        "src/net.cpp",
        "src/wallet/wallet.cpp",
    ]),

    ("django", "django", "main", [
        "django/core/handlers/base.py",
        "django/db/models/base.py",
        "django/views/generic/base.py",
        "django/contrib/admin/options.py",
        "django/forms/models.py",
        "django/urls/resolvers.py",
    ]),
    ("pandas-dev", "pandas", "main", [
        "pandas/core/frame.py",
        "pandas/core/series.py",
    ]),

    ("vuejs", "core", "main", [
        "packages/runtime-core/src/renderer.ts",
        "packages/reactivity/src/reactive.ts",
    ]),
    ("tailwindlabs", "tailwindcss", "next", [
        "packages/@tailwindcss-node/src/index.ts",
    ]),

    ("golang", "go", "master", [
        "src/net/http/server.go",
        "src/fmt/print.go",
    ]),
    ("flutter", "flutter", "master", [
        "packages/flutter/lib/src/widgets/framework.dart",
    ]),
    ("JetBrains", "kotlin", "master", [
        "compiler/frontend/src/org/jetbrains/kotlin/resolve/BindingContext.kt",
    ]),
    ("ziglang", "zig", "master", [
        "lib/std/mem.zig",
    ]),
    ("elixir-lang", "elixir", "main", [
        "lib/elixir/lib/enum.ex",
    ]),
    ("apache", "spark", "master", [
        "core/src/main/scala/org/apache/spark/SparkContext.scala",
    ]),
    ("apache", "kafka", "trunk", [
        "core/src/main/scala/kafka/server/KafkaServer.scala",
    ]),
    ("hashicorp", "vault", "main", [
        "vault/logical_system.go",
    ]),
    ("hashicorp", "terraform", "main", [
        "internal/command/apply.go",
    ]),
    ("prometheus", "prometheus", "main", [
        "main.go",
    ]),
    ("grafana", "grafana", "main", [
        "pkg/api/api.go",
    ]),
    ("microsoft", "DeepSpeed", "main", [
        "deepspeed/runtime/engine.py",
        "deepspeed/ops/adam/cpu_adam.py",
    ]),
    ("NVIDIA", "Megatron-LM", "main", [
        "megatron/training.py",
        "megatron/model/transformer.py",
    ]),
    ("Significant-Gravitas", "AutoGPT", "master", [
        "autogpt/agent/agent.py",
    ]),
    ("google", "jax", "main", [
        "jax/_src/core.py",
    ]),
    ("Lightning-AI", "pytorch-lightning", "master", [
        "src/lightning/pytorch/trainer/trainer.py",
    ]),
    ("QwikDev", "qwik", "main", [
        "packages/qwik/src/core/render/render.public.ts",
    ]),
    ("honojs/hono", "hono", "main", [
        "src/hono.ts",
    ]),
    ("vercel", "turborepo", "main", [
        "packages/turbo-lib/src/run/mod.rs",
    ]),
    ("TanStack", "query", "main", [
        "packages/query-core/src/queryClient.ts",
    ]),
    ("pola-rs", "polars", "main", [
        "crates/polars-core/src/frame/mod.rs",
    ]),
    ("surrealdb", "surrealdb", "main", [
        "src/kvs/mod.rs",
    ]),
    ("meilisearch", "meilisearch", "main", [
        "meilisearch/src/main.rs",
    ]),
    ("diesel-rs", "diesel", "master", [
        "diesel/src/lib.rs",
    ]),
    ("pocketbase", "pocketbase", "main", [
        "core/app.go",
    ]),
    ("go-gitea", "gitea", "main", [
        "models/repo/repo.go",
    ]),
    ("rclone", "rclone", "master", [
        "cmd/rclone.go",
    ]),
    ("traefik", "traefik", "master", [
        "pkg/server/server.go",
    ]),
    ("opentofu", "opentofu", "main", [
        "internal/command/plan.go",
    ]),
    ("localstack", "localstack", "main", [
        "localstack/services/s3/models.py",
    ]),
    ("cilium", "cilium", "main", [
        "pkg/agent/main.go",
    ]),
    ("istio", "istio", "master", [
        "pilot/pkg/model/config.go",
    ]),
    ("dapr", "dapr", "master", [
        "pkg/runtime/runtime.go",
    ]),
    ("shadcn-ui", "ui", "main", [
        "apps/www/registry/default/ui/button.tsx",
    ]),
    ("ratatui-org", "ratatui", "main", [
        "src/terminal.rs",
        "src/widgets/paragraph.rs",
        "src/layout.rs",
        "src/buffer.rs",
    ]),
    ("FlowiseAI", "Flowise", "main", [
        "packages/server/src/index.ts",
        "packages/components/nodes/LLMs/OpenAI/OpenAI.ts",
        "packages/components/nodes/Chains/LLMChain/LLMChain.ts",
    ]),
    ("ai16z", "eliza", "main", [
        "src/core/agent.ts",
        "src/core/context.ts",
        "src/services/social.ts",
    ]),
    ("volcengine", "verl", "main", [
        "verl/trainer/ppo.py",
        "verl/models/llama.py",
    ]),
    ("OpenGVLab", "LightRAG", "main", [
        "lightrag/core/model.py",
        "lightrag/core/retriever.py",
    ]),
    ("anthropics", "anthropic-sdk-python", "main", [
        "src/anthropic/resources/messages.py",
        "src/anthropic/_client.py",
    ]),
    ("openai", "openai-python", "main", [
        "src/openai/resources/chat/completions.py",
        "src/openai/_client.py",
    ]),
    ("google", "flax", "main", [
        "flax/linen/linear.py",
        "flax/training/train_state.py",
    ]),
    ("keras-team", "keras", "master", [
        "keras/layers/core/dense.py",
        "keras/models/model.py",
        "keras/optimizers/adam.py",
    ]),
    ("facebookresearch", "llama", "main", [
        "llama/model.py",
        "llama/tokenizer.py",
        "llama/generation.py",
    ]),
    ("honojs", "hono", "main", [
        "src/hono.ts",
        "src/router.ts",
        "src/context.ts",
        "src/middleware/logger/index.ts",
    ]),
    ("redwoodjs", "redwood", "main", [
        "packages/api/src/functions/graphql.ts",
        "packages/router/src/router.tsx",
    ]),
    ("TanStack", "router", "main", [
        "packages/react-router/src/router.ts",
        "packages/react-router/src/route.ts",
    ]),
    ("TanStack", "form", "main", [
        "packages/react-form/src/useForm.tsx",
    ]),
    ("tldraw", "tldraw", "main", [
        "packages/tldraw/src/lib/Tldraw.tsx",
        "packages/editor/src/lib/editor/Editor.ts",
    ]),
    ("facebook", "lexical", "main", [
        "packages/lexical/src/LexicalEditor.ts",
        "packages/lexical-react/src/LexicalComposer.tsx",
    ]),
    ("shadcn", "taxonomy", "main", [
        "app/(marketing)/layout.tsx",
        "config/site.ts",
    ]),
    ("rustdesk", "rustdesk", "master", [
        "src/main.rs",
        "src/server.rs",
        "src/client.rs",
    ]),
    ("awesome-selfhosted", "awesome-selfhosted", "master", [
        "README.md",
    ]),
    ("n8n-io", "n8n", "master", [
        "packages/cli/src/Server.ts",
        "packages/nodes-base/nodes/HttpRequest/HttpRequest.node.ts",
    ]),
    ("AUTOMATIC1111", "stable-diffusion-webui", "master", [
        "modules/ui.py",
        "modules/processing.py",
        "launch.py",
    ]),
    ("nodejs", "node", "main", [
        "lib/internal/fs/utils.js",
        "src/node.cc",
        "deps/v8/src/api/api.cc",
    ]),
    ("denoland", "deno", "main", [
        "cli/main.rs",
        "runtime/worker.rs",
    ]),
    ("llvm", "llvm-project", "main", [
        "llvm/lib/IR/Core.cpp",
        "clang/lib/AST/ASTContext.cpp",
    ]),
    ("docker", "cli", "master", [
        "cli/command/container/run.go",
    ]),
    ("docker", "compose", "main", [
        "pkg/compose/service.go",
    ]),
    ("hashicorp", "consul", "main", [
        "agent/consul/server.go",
    ]),
    ("hashicorp", "nomad", "main", [
        "nomad/client.go",
    ]),
    ("hashicorp", "packer", "main", [
        "packer/builder.go",
    ]),
    ("prometheus", "alertmanager", "main", [
        "dispatch/dispatch.go",
    ]),
    ("grafana", "loki", "main", [
        "pkg/ingester/ingester.go",
    ]),
    ("grafana", "tempo", "main", [
        "pkg/tempo/tempo.go",
    ]),
    ("elastic", "kibana", "main", [
        "src/core/server/server.ts",
    ]),
    ("elastic", "logstash", "main", [
        "logstash-core/lib/logstash/runner.rb",
    ]),
    ("postgres", "postgres", "master", [
        "src/backend/executor/execMain.c",
        "src/backend/storage/buffer/bufmgr.c",
    ]),
    ("mongodb", "mongo", "master", [
        "src/mongo/db/query/executor.cpp",
    ]),
    ("yugabyte", "yugabyte-db", "master", [
        "src/yb/tablet/tablet.cc",
    ]),
    ("scylladb", "scylladb", "master", [
        "main.cc",
    ]),
    ("qdrant", "qdrant", "master", [
        "lib/collection/src/collection.rs",
    ]),
    ("chroma-core", "chroma", "main", [
        "chromadb/api/segment.py",
    ]),
    ("weaviate", "weaviate", "master", [
        "entities/models/object.go",
    ]),
    ("dask", "dask", "main", [
        "dask/array/core.py",
        "dask/dataframe/core.py",
    ]),
    ("mui", "material-ui", "master", [
        "packages/mui-material/src/Button/Button.js",
    ]),
    ("ant-design", "ant-design", "master", [
        "components/button/index.tsx",
    ]),
    ("element-plus", "element-plus", "dev", [
        "packages/components/button/src/button.vue",
    ]),
    ("ionic-team", "ionic-framework", "main", [
        "core/src/components/button/button.tsx",
    ]),
    ("flutter", "flutter", "master", [
        "packages/flutter/lib/src/widgets/framework.dart",
        "packages/flutter/lib/src/material/button.dart",
        "packages/flutter/lib/src/cupertino/button.dart",
    ]),
    ("numpy", "numpy", "main", [
        "numpy/core/src/multiarray/arrayobj.c",
        "numpy/fft/_pocketfft.py",
    ]),
    ("scipy", "scipy", "main", [
        "scipy/optimize/_minimize.py",
    ]),
    ("matplotlib", "matplotlib", "main", [
        "lib/matplotlib/axes/_axes.py",
    ]),
    ("plotly", "plotly.js", "master", [
        "src/core.js",
    ]),
    ("d3", "d3", "main", [
        "src/index.js",
    ]),
    ("threejs", "three.js", "dev", [
        "src/core/Object3D.js",
        "src/renderers/WebGLRenderer.js",
    ]),
    ("ethereum", "go-ethereum", "master", [
        "eth/backend.go",
        "core/state_processor.go",
    ]),
    ("solana-labs", "solana", "master", [
        "runtime/src/bank.rs",
    ]),
    ("near", "nearcore", "master", [
        "chain/chain/src/chain.rs",
    ]),
    ("paritytech", "polkadot-sdk", "master", [
        "polkadot/node/service/src/lib.rs",
    ]),
    ("aptos-labs", "aptos-core", "main", [
        "aptos-node/src/main.rs",
    ]),
    ("sui-foundation/sui", "sui", "main", [
        "crates/sui-node/src/lib.rs",
    ]),
    ("cosmos", "cosmos-sdk", "main", [
        "baseapp/baseapp.go",
    ]),
    ("AUTOMATIC1111", "stable-diffusion-webui", "master", [
        "modules/txt2img.py",
        "modules/img2img.py",
    ]),
    ("opencv", "opencv", "4.x", [
        "modules/imgproc/src/color.cpp",
        "modules/core/src/matrix.cpp",
    ]),
    ("mistralai", "mistral-common", "main", [
        "src/mistral_common/tokens/tokenizers/base.py",
    ]),
    ("ollama", "ollama-python", "main", [
        "ollama/_client.py",
    ]),
    ("comfyanonymous", "ComfyUI", "master", [
        "main.py",
        "nodes.py",
        "execution.py",
    ]),
    ("RVC-Boss", "GPT-SoVITS", "main", [
        "GPT_SoVITS/inference_webui.py",
    ]),
    ("ZDR0", "KAG", "main", [
        "kag/common/base.py",
    ]),
    ("mem0ai", "mem0", "main", [
        "mem0/core/main.py",
    ]),
    ("agno-ai", "agno", "main", [
        "agno/agent/agent.py",
    ]),
    ("shadcn-ui", "taxonomy", "main", [
        "components/main-nav.tsx",
    ]),
    ("pmndrs", "react-three-fiber", "master", [
        "packages/fiber/src/core/loop.ts",
    ]),
    ("framer", "motion", "main", [
        "packages/framer-motion/src/motion/index.tsx",
    ]),
    ("facebook", "stylex", "main", [
        "packages/stylex/src/stylex.js",
    ]),
    ("expo", "expo", "main", [
        "packages/expo/src/Expo.ts",
    ]),
    ("facebook", "react-native", "main", [
        "ReactAndroid/src/main/java/com/facebook/react/ReactRootView.java",
    ]),
    ("desktop", "desktop", "development", [
        "app/src/lib/git/core.ts",
    ]),
    ("projectdiscovery", "nuclei", "main", [
        "pkg/protocols/http/http.go",
    ]),
    ("rapid7", "metasploit-framework", "master", [
        "lib/msf/core/exploit.rb",
    ]),
    ("sqlmapproject", "sqlmap", "master", [
        "sqlmap.py",
    ]),
    ("freeCodeCamp", "freeCodeCamp", "main", [
        "curriculum/challenges/english/01-responsive-web-design/basic-html-and-html5/say-hello-to-html-elements.md",
        "api/src/server.ts",
    ]),
    ("TheAlgorithms", "Python", "master", [
        "machine_learning/decision_tree.py",
        "neural_network/back_propagation.py",
    ]),
    ("donnemartin", "system-design-primer", "master", [
        "solutions/system_design/social_graph/README.md",
    ]),
    ("kamranahmedse", "developer-roadmap", "master", [
        "src/components/Roadmap.tsx",
    ]),
    ("public-apis", "public-apis", "master", [
        "scripts/validate.py",
    ]),
    ("EbookFoundation", "free-programming-books", "main", [
        "free-programming-books-es.md",
    ]),
    ("ohmyzsh", "ohmyzsh", "master", [
        "oh-my-zsh.sh",
        "plugins/git/git.plugin.zsh",
    ]),
    ("home-assistant", "core", "dev", [
        "homeassistant/core.py",
        "homeassistant/components/light/__init__.py",
    ]),
    ("oscipv", "Cline", "main", [
        "src/core/Cline.ts",
    ]),
    ("TabbyML", "tabby", "main", [
        "crates/tabby/src/main.rs",
    ]),
    ("drizzle-team", "drizzle-orm", "main", [
        "packages/drizzle-orm/src/pg-core/table.ts",
    ]),
    ("oven-sh", "bun", "main", [
        "src/main.zig",
        "src/bun.js/bindings/BunClient.cpp",
    ]),
    ("refined-github", "refined-github", "main", [
        "source/refined-github.ts",
    ]),
    ("jwasham", "coding-interview-university", "main", [
        "translations/README-es.md",
    ]),
    ("trekhleb", "javascript-algorithms", "master", [
        "src/algorithms/math/fibonacci/fibonacci.js",
    ]),
    ("ryanmcdermott", "clean-code-javascript", "master", [
        "README.md",
    ]),
    ("goldbergyoni", "nodebestpractices", "master", [
        "README.md",
    ]),
    ("sindresorhus", "awesome", "main", [
        "readme.md",
    ]),
    ("vinta", "awesome-python", "master", [
        "README.md",
    ]),
    ("avelino", "awesome-go", "master", [
        "README.md",
    ]),
    ("rust-unofficial", "awesome-rust", "main", [
        "README.md",
    ]),
    ("josephmisiti", "awesome-machine-learning", "master", [
        "README.md",
    ]),
    ("enaqx", "awesome-pentest", "master", [
        "README.md",
    ]),
    ("Solido", "awesome-flutter", "master", [
        "README.md",
    ]),
    ("dzharii", "awesome-typescript", "master", [
        "README.md",
    ]),
     # Proyectos educativos en español
    ("mouredev", "Hello-Python", "main", [
        "README.md",
        "Topics/01_introduction.py",
        "Topics/02_variables.py",
        "Topics/03_data_types.py",
    ]),
    ("mouredev", "roadmap-retos-programacion", "main", [
        "Roadmap/README.md",
    ]),
    ("midudev", "apuntes-de-javascript", "main", [
        "README.md",
    ]),

    ("google", "ml-foundational-courses", "main", [
        "machine_learning_crash_course/README.md",
    ]),
    ("rasbt", "LLMs-from-scratch", "main", [
        "ch02/01_main-chapter-code/ch02.py",
        "ch03/01_main-chapter-code/ch03.py",
        "ch04/01_main-chapter-code/ch04.py",
        "ch05/01_main-chapter-code/ch05.py",
        "appendix-D/01_main-chapter-code/appendix-D.py",
    ]),
    ("karpathy", "makemore", "master", [
        "makemore.py",
        "bigram.py",
    ]),
    ("karpathy", "micrograd", "master", [
        "micrograd/engine.py",
        "micrograd/nn.py",
        "demo.ipynb",
    ]),
    ("google-deepmind", "gemma", "main", [
        "gemma/transformer.py",
        "gemma/modules.py",
        "gemma/sampler.py",
    ]),
    ("meta-llama", "llama-stack", "main", [
        "llama_stack/models/llama/prompt_templates.py",
    ]),

    ("pydantic", "pydantic-ai", "main", [
        "pydantic_ai_slim/pydantic_ai/agent.py",
        "pydantic_ai_slim/pydantic_ai/models/openai.py",
    ]),
    ("microsoft", "autogen", "main", [
        "python/packages/autogen-agentchat/autogen_agentchat/agents/assistant_agent.py",
    ]),
    ("BerriAI", "litellm", "main", [
        "litellm/main.py",
        "litellm/utils.py",
    ]),
    ("openai", "tiktoken", "main", [
        "tiktoken/core.py",
        "tiktoken/_educational.py",
    ]),
    ("jlowin", "fastmcp", "main", [
        "src/fastmcp/server.py",
        "src/fastmcp/tools/tool_manager.py",
    ]),
    ("encode", "httpx", "master", [
        "httpx/_client.py",
        "httpx/_models.py",
        "httpx/_transports/default.py",
    ]),
    ("encode", "starlette", "master", [
        "starlette/applications.py",
        "starlette/routing.py",
        "starlette/middleware/cors.py",
    ]),
    ("celery", "celery", "main", [
        "celery/app/base.py",
        "celery/worker/worker.py",
        "celery/backends/redis.py",
    ]),

    ("TheAlgorithms", "JavaScript", "master", [
        "Data-Structures/Linked-List/SinglyLinkedList.js",
        "Sorts/BubbleSort.js",
        "Sorts/QuickSort.js",
        "Graphs/Dijkstra.js",
    ]),
    ("TheAlgorithms", "Java", "master", [
        "src/main/java/com/thealgorithms/sorts/QuickSort.java",
        "src/main/java/com/thealgorithms/datastructures/trees/BinaryTree.java",
        "src/main/java/com/thealgorithms/dynamicprogramming/Fibonacci.java",
    ]),
    ("TheAlgorithms", "C-Plus-Plus", "master", [
        "sorting/quick_sort.cpp",
        "data_structures/linked_list.cpp",
        "graph/dijkstra.cpp",
        "dynamic_programming/fibonacci.cpp",
    ]),
    ("TheAlgorithms", "Rust", "master", [
        "src/sorting/quick_sort.rs",
        "src/data_structures/linked_list.rs",
        "src/graph/dijkstra.rs",
    ]),

    ("argoproj", "argo-workflows", "main", [
        "workflow/controller/workflowpod.go",
    ]),
    ("open-telemetry", "opentelemetry-python", "main", [
        "opentelemetry-api/src/opentelemetry/trace/__init__.py",
        "opentelemetry-sdk/src/opentelemetry/sdk/trace/__init__.py",
    ]),
    ("encode", "uvicorn", "master", [
        "uvicorn/main.py",
        "uvicorn/workers.py",
        "uvicorn/protocols/http/h11_impl.py",
    ]),
    ("vercel", "next.js", "canary", [
        "packages/next/src/server/next-server.ts",
        "packages/next/src/client/router.ts",
    ]),
    ("sveltejs", "svelte", "main", [
        "packages/svelte/src/compiler/compile/index.js",
    ]),
    ("python", "cpython", "main", [
        "Objects/dictobject.c",
        "Objects/listobject.c",
        "Python/ceval.c",
    ]),
    ("pytorch", "vision", "main", [
        "torchvision/models/resnet.py",
        "torchvision/transforms/transforms.py",
    ]),
    ("huggingface", "diffusers", "main", [
        "src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py",
        "src/diffusers/models/unet_2d_condition.py",
    ]),
    ("langchain-ai", "langchain", "master", [
        "libs/langchain/langchain/chains/llm.py",
        "libs/langchain/langchain/agents/agent.py",
    ]),
    ("hwchase17", "langchainjs", "main", [
        "langchain/src/chains/llm_chain.ts",
        "langchain/src/agents/agent.ts",
    ]),
    ("elastic", "elasticsearch", "main", [
        "server/src/main/java/org/elasticsearch/search/SearchService.java",
        "server/src/main/java/org/elasticsearch/cluster/ClusterState.java",
    ]),
    ("apache", "spark", "master", [
        "sql/core/src/main/scala/org/apache/spark/sql/DataFrame.scala",
        "core/src/main/scala/org/apache/spark/rdd/RDD.scala",
    ]),
    ("apache", "kafka", "trunk", [
        "core/src/main/scala/kafka/server/KafkaServer.scala",
        "clients/src/main/java/org/apache/kafka/clients/producer/KafkaProducer.java",
    ]),
    ("neo4j", "neo4j", "dev", [
        "community/kernel/src/main/java/org/neo4j/kernel/impl/core/NodeManager.java",
    ]),
    ("psycopg", "psycopg", "master", [
        "psycopg/psycopg/connection.py",
        "psycopg/psycopg/cursor.py",
    ]),
    ("pallets", "flask", "main", [
        "src/flask/app.py",
        "src/flask/blueprints.py",
    ]),
    ("tiangolo", "fastapi", "master", [
        "fastapi/applications.py",
        "fastapi/routing.py",
    ]),
    ("django", "django", "main", [
        "django/core/handlers/base.py",
        "django/db/models/query.py",
    ]),
    ("rails", "rails", "main", [
        "activerecord/lib/active_record/base.rb",
        "actionpack/lib/action_controller/metal.rb",
    ]),
    ("laravel", "framework", "master", [
        "src/Illuminate/Foundation/Application.php",
        "src/Illuminate/Database/Eloquent/Model.php",
    ]),
    ("symfony", "symfony", "7.1", [
        "src/Symfony/Component/HttpKernel/HttpKernel.php",
        "src/Symfony/Component/Routing/Router.php",
    ]),
    ("spring-projects", "spring-boot", "main", [
        "spring-boot-project/spring-boot/src/main/java/org/springframework/boot/SpringApplication.java",
    ]),
    ("microsoft", "playwright", "main", [
        "packages/playwright-core/src/server/browser.ts",
        "packages/playwright-core/src/server/page.ts",
    ]),
    ("UnslothAI", "unsloth", "main", [
        "unsloth/models/llama.py",
    ]),
    ("binary-husky", "gpt_academic", "master", [
        "main.py",
    ]),
    ("OpenGVLab", "InternVL", "main", [
        "internvl_chat/modeling_internvl_chat.py",
    ]),
    ("deepseek-ai", "DeepSeek-V2", "main", [
        "deepseek_v2/modeling_deepseek.py",
    ]),
    ("Soulter", "AstrBot", "main", [
        "main.py",
    ]),
    ("assafelovic", "gpt-researcher", "master", [
        "gpt_researcher/master/agent.py",
    ]),
    ("argoproj", "argo-cd", "master", [
        "util/db/db.go",
    ]),
    ("rancher", "rancher", "master", [
        "pkg/main.go",
    ]),
    ("crossplane", "crossplane", "main", [
        "pkg/main.go",
    ]),
    ("knative", "serving", "main", [
        "pkg/main.go",
    ]),
    ("apache", "airflow", "main", [
        "airflow/models/dag.py",
    ]),
    ("fish-shell", "fish-shell", "master", [
        "src/fish.cpp",
    ]),
    ("astral-sh", "uv", "main", [
        "crates/uv/src/main.rs",
    ]),
    ("t3-oss", "create-t3-app", "main", [
        "src/index.ts",
    ]),
    ("yt-dlp", "yt-dlp", "master", [
        "yt_dlp/YoutubeDL.py",
    ]),
    ("tldr-pages", "tldr", "main", [
        "README.md",
    ]),
    ("hashicorp", "boundary", "main", [
        "api/client.go",
    ]),
    ("hashicorp", "waypoint", "main", [
        "builtin/docker/builder.go",
    ]),
    ("minio", "minio", "master", [
        "cmd/server-main.go",
    ]),
    ("envoyproxy", "envoy", "main", [
        "source/common/common/base64.cc",
    ]),
    ("appwrite", "appwrite", "master", [
        "app/controllers/api/users.php",
    ]),
    ("directus", "directus", "main", [
        "api/src/app.ts",
    ]),
    ("strapi", "strapi", "main", [
        "packages/core/strapi/lib/Strapi.js",
    ]),
    ("ghost", "Ghost", "main", [
        "core/server/api/v2/posts.js",
    ]),
    ("zaproxy", "zaproxy", "main", [
        "src/org/zaproxy/zap/ZAP.java",
    ]),
    ("wireshark", "wireshark", "master", [
        "epan/packet.c",
    ]),
    ("aircrack-ng", "aircrack-ng", "master", [
        "src/aircrack-ng.c",
    ]),
    ("EmpireProject", "Empire", "master", [
        "lib/common/helpers.py",
    ]),
    ("crystal-lang", "crystal", "master", [
        "src/compiler/crystal/compiler.cr",
    ]),
    ("julialang", "julia", "master", [
        "src/base.jl",
    ]),
    ("nim-lang", "nim", "devel", [
        "compiler/nim.nim",
    ]),
    ("gleam-lang", "gleam", "main", [
        "compiler-core/src/lib.rs",
    ]),
    ("racket", "racket", "master", [
        "racket/src/racket/main.c",
    ]),
    ("ocaml", "ocaml", "trunk", [
        "runtime/main.c",
    ]),
    ("clojure", "clojure", "master", [
        "src/jvm/clojure/lang/RT.java",
    ]),
    ("bevyengine", "bevy", "main", [
        "crates/bevy_ecs/src/lib.rs",
    ]),
    ("livebook-dev", "livebook", "main", [
        "lib/livebook.ex",
    ]),
    ("phoenixframework", "phoenix", "master", [
        "lib/phoenix.ex",
    ]),
    ("absinthe-graphql", "absinthe", "master", [
        "lib/absinthe.ex",
    ]),
    ("nerves-project", "nerves", "master", [
        "lib/nerves.ex",
    ]),
    ("exercism", "exercism", "main", [
        "README.md",
    ]),
    ("the-roadmap", "roadmap", "master", [
        "README.md",
    ]),
    ("ossu", "computer-science", "master", [
        "README.md",
    ]),
    ("news-ycombinator", "news-ycombinator.github.io", "master", [
        "README.md",
    ]),
    ("build-your-own-x", "build-your-own-x", "master", [
        "README.md",
    ]),
    ("papers-we-love", "papers-we-love", "master", [
        "README.md",
    ]),
    ("reddit-archive", "reddit", "master", [
        "r2/r2/models/post.py",
    ]),
    ("mastodon", "mastodon", "main", [
        "app/models/status.rb",
    ]),
    ("lemmy-net", "lemmy", "main", [
        "src/main.rs",
    ]),
    ("matrix-org", "synapse", "develop", [
        "synapse/storage/databases/main/events.py",
    ]),
    ("996icu", "996.ICU", "main", []),
    ("vuejs", "vue", "main", []),
    ("trimstray", "the-book-of-secret-knowledge", "main", []),
    ("tensorflow", "tensorflow", "main", []),
    ("getify", "You-Dont-Know-JS", "main", []),
    ("airbnb", "javascript", "main", []),
    ("github", "gitignore", "main", []),
    ("twbs", "bootstrap", "main", []),
    ("521xueweihan", "HelloGitHub", "main", []),
    ("javascript-tutorial", "en.javascript.info", "main", []),
    ("developer-can", "can-i-use", "main", []),
    ("cycfi", "elements", "main", []),
    ("yangshun", "tech-interview-handbook", "main", []),
    ("Genymobile", "scrcpy", "main", []),
    ("labuladong", "fucking-algorithm", "main", []),
    ("langgenius", "dify", "main", []),
    ("microsoft", "PowerToys", "main", []),
    ("Chalarangelo", "30-seconds-of-code", "main", []),
    ("electron", "electron", "main", []),
    ("ruanyf", "weekly", "main", []),
    ("3b1b", "manim", "main", []),
    ("microsoft", "ML-For-Beginners", "main", []),
    ("louislam", "uptime-kuma", "main", []),
    ("laravel", "laravel", "main", []),
    ("MunGell", "awesome-for-beginners", "main", []),
    ("macrozheng", "mall", "main", []),
    ("Snailclimb", "JavaGuide", "main", []),
    ("Hack-with-Github", "Awesome-Hacking", "main", []),
    ("microsoft", "generative-ai-for-beginners", "main", []),
    ("godotengine", "godot", "main", []),
    ("fatedier", "frp", "main", []),
    ("tiimgreen", "github-cheat-sheet", "main", []),
    ("dylanaraps", "neofetch", "main", []),
    ("shadowsocks", "shadowsocks-windows", "main", []),
    ("axios", "axios", "main", []),
    ("NVlabs", "stylegan", "main", []),
    ("rby87", "docker-compose-examples", "main", []),
    ("mrdoob", "three.js", "main", []),
    ("reduxjs", "redux", "main", []),
    ("webpack", "webpack", "main", []),
    ("ansible", "ansible", "main", []),
    ("ohmyzsh", "oh-my-zsh", "main", []),
    ("h5bp", "html5-boilerplate", "main", []),
    ("FortAwesome", "Font-Awesome", "main", []),
    ("animate-css", "animate.css", "main", []),
    ("jquery", "jquery", "main", []),
    ("lodash", "lodash", "main", []),
    ("moment", "moment", "main", []),
    ("chartjs", "Chart.js", "main", []),
    ("ghost", "ghost", "main", []),
    ("WordPress", "WordPress", "main", []),
    ("jekyll", "jekyll", "main", []),
    ("hexojs", "hexo", "main", []),
    ("Automattic", "mongoose", "main", []),
    ("socketio", "socket.io", "main", []),
    ("nestjs", "nest", "main", []),
    ("koajs", "koa", "main", []),
    ("meteor", "meteor", "main", []),
    ("renpy", "renpy", "main", []),
    ("love2d", "love", "main", []),
    ("pixijs", "pixijs", "main", []),
    ("phaserjs", "phaser", "main", []),
    ("defold", "defold", "main", []),
    ("libgdx", "libgdx", "main", []),
    ("helm", "helm", "main", []),
    ("open-policy-agent", "opa", "main", []),
    ("nginx", "nginx", "main", []),
    ("apache", "httpd", "main", []),
    ("caddyserver", "caddy", "main", []),
    ("valkey-io", "valkey", "main", []),
    ("apache", "flink", "main", []),
    ("apache", "cassandra", "main", []),
    ("apache", "hadoop", "main", []),
    ("apache", "zookeeper", "main", []),
    ("apache", "dubbo", "main", []),
    ("apache", "rocketmq", "main", []),
    ("apache", "pulsar", "main", []),
    ("apache", "druid", "main", []),
    ("apache", "superset", "main", []),
    ("apache", "nifi", "main", []),
    ("apache", "beam", "main", []),
    ("apache", "thrift", "main", []),
    ("grpc", "grpc", "main", []),
    ("protocolbuffers", "protobuf", "main", []),
    ("etcd-io", "etcd", "main", []),
    ("coredns", "coredns", "main", []),
    ("containerd", "containerd", "main", []),
    ("opencontainers", "runc", "main", []),
    ("kata-containers", "kata-containers", "main", []),
    ("firecracker-microvm", "firecracker", "main", []),
    ("typesense", "typesense", "main", []),
    ("pinecone-io", "pinecone-python-client", "main", []),
    ("openai", "whisper", "main", []),
    ("openai", "CLIP", "main", []),
    ("openai", "gpt-2", "main", []),
    ("facebookresearch", "segment-anything", "main", []),
    ("facebookresearch", "detectron2", "main", []),
    ("facebookresearch", "fastText", "main", []),
    ("google-research", "bert", "main", []),
    ("google-research", "t5x", "main", []),
    ("google", "guava", "main", []),
    ("google", "googletest", "main", []),
    ("google", "flatbuffers", "main", []),
    ("google", "brotli", "main", []),
    ("deepmind", "alphafold", "main", []),
    ("huggingface", "datasets", "main", []),
    ("huggingface", "tokenizers", "main", []),
    ("huggingface", "accelerate", "main", []),
    ("Stability-AI", "StableDiffusion", "main", []),
    ("CompVis", "stable-diffusion", "main", []),
    ("runwayml", "stable-diffusion-v1-5", "main", []),
    ("localai", "LocalAI", "main", []),
    ("lmstudio-ai", "lmstudio", "main", []),
    ("tgi-huggingface", "text-generation-inference", "main", []),
    ("mistralai", "mistral-src", "main", []),
    ("databricks", "dolly", "main", []),
    ("mosaicml", "llm-foundry", "main", []),
    ("replicate", "cog", "main", []),
    ("pytorch", "serve", "main", []),
    ("triton-inference-server", "server", "main", []),
    ("onnx", "onnx", "main", []),
    ("microsoft", "onnxruntime", "main", []),
    ("microsoft", "semantic-kernel", "main", []),
    ("microsoft", "LightGBM", "main", []),
    ("microsoft", "terminal", "main", []),
    ("microsoft", "playwright-python", "main", []),
    ("microsoft", "fluentui", "main", []),
    ("microsoft", "cascadia-code", "main", []),
    ("microsoft", "calculator", "main", []),
    ("microsoft", "WSL", "main", []),
    ("microsoft", "Windows-driver-samples", "main", []),
    ("microsoft", "winget-pkgs", "main", []),
    ("microsoft", "vcpkg", "main", []),
    ("microsoft", "dotnet", "main", []),
    ("microsoft", "aspnetcore", "main", []),
    ("microsoft", "entityframeworkcore", "main", []),
    ("microsoft", "TypeScript-Handbook", "main", []),
    ("microsoft", "GSL", "main", []),
    ("microsoft", "DirectX-Graphics-Samples", "main", []),
    ("microsoft", "Azure-Samples", "main", []),
    ("microsoft", "ML.NET", "main", []),
    ("microsoft", "qsharp-compiler", "main", []),
    ("apple", "swift", "main", []),
    ("apple", "swift-algorithms", "main", []),
    ("apple", "swift-nio", "main", []),
    ("apple", "foundationdb", "main", []),
    ("apple", "cups", "main", []),
    ("apple", "darwin-xnu", "main", []),
    ("apple", "swift-syntax", "main", []),
    ("apple", "swift-package-manager", "main", []),
    ("apple", "homekit-adk", "main", []),
    ("apple", "ml-stable-diffusion", "main", []),
    ("apple", "axlearn", "main", []),
    ("amazon-archives", "amazon-sagemaker-examples", "main", []),
    ("aws", "aws-cdk", "main", []),
    ("aws", "aws-cli", "main", []),
    ("aws", "aws-sdk-js", "main", []),
    ("aws", "aws-lambda-powertools-python", "main", []),
    ("aws", "amazon-sashimi", "main", []),
    ("aws", "amazon-vpc-cni-k8s", "main", []),
    ("aws", "firecracker-microvm", "main", []),
    ("aws", "sagemaker-python-sdk", "main", []),
    ("aws", "chalice", "main", []),
    ("aws-samples", "aws-builders-library-samples", "main", []),
    ("google", "material-design-icons", "main", []),
    ("google", "fonts", "main", []),
    ("google", "web-starter-kit", "main", []),
    ("google", "zx", "main", []),
    ("google", "wire", "main", []),
    ("google", "grumpy", "main", []),
    ("google", "sanitizers", "main", []),
    ("google", "tink", "main", []),
    ("google", "cadvisor", "main", []),
    ("google", "skia", "main", []),
    ("google", "angle", "main", []),
    ("google", "leveldb", "main", []),
    ("google", "re2", "main", []),
    ("google", "snappy", "main", []),
    ("google", "highwayhash", "main", []),
    ("google", "farmhash", "main", []),
    ("google", "cityhash", "main", []),
    ("google", "double-conversion", "main", []),
    ("google", "benchmark", "main", []),
    ("google", "draco", "main", []),
    ("google", "filament", "main", []),
    ("google", "v8", "main", []),
    ("google", "dart-sdk", "main", []),
    ("google", "flutter-desktop-embedding", "main", []),
    ("google", "fuchsia", "main", []),
    ("google", "bloaty", "main", []),
    ("google", "perfetto", "main", []),
    ("google", "os-login", "main", []),
    ("google", "identity-toolkit-python", "main", []),
    ("google", "oauth2client", "main", []),
    ("google", "google-api-python-client", "main", []),
    ("google", "google-cloud-python", "main", []),
    ("google", "closure-compiler", "main", []),
    ("google", "closure-library", "main", []),
    ("google", "blockly", "main", []),
    ("google", "shaka-player", "main", []),
    ("google", "shaka-streamer", "main", []),
    ("google", "model-viewer", "main", []),
    ("google", "mediapipe", "main", []),
    ("google", "orb-slam3", "main", []),
    ("google", "cartographer", "main", []),
    ("google", "maglev", "main", []),
    ("google", "sea-orm", "main", []),
    ("google", "sqlx", "main", []),
    ("google", "tokio", "main", []),
    ("google", "hyper", "main", []),
    ("google", "reqwest", "main", []),
    ("google", "serde", "main", []),
    ("google", "syn", "main", []),
    ("google", "quote", "main", []),
    ("google", "proc-macro2", "main", []),
    ("google", "libc", "main", []),
    ("google", "log", "main", []),
    ("google", "env_logger", "main", []),
    ("google", "anyhow", "main", []),
    ("google", "thiserror", "main", []),
    ("google", "rand", "main", []),
    ("google", "regex", "main", []),
    ("google", "lazy_static", "main", []),
    ("google", "bitflags", "main", []),
    ("google", "cfg-if", "main", []),
    ("google", "smallvec", "main", []),
    ("google", "parking_lot", "main", []),
    ("google", "crossbeam", "main", []),
    ("google", "rayon", "main", []),
    ("google", "itertools", "main", []),
    ("google", "num", "main", []),
    ("google", "chrono", "main", []),
    ("google", "url", "main", []),
    ("google", "percent-encoding", "main", []),
    ("google", "idna", "main", []),
    ("google", "unicode-normalization", "main", []),
    ("google", "unicode-bidi", "main", []),
    ("google", "unicode-width", "main", []),
    ("google", "unicode-segmentation", "main", []),
    ("google", "unicode-case-mapping", "main", []),
    ("gto76", "python-cheatsheet", "main", []),
    ("PhilJay", "MPAndroidChart", "main", []),
    ("lapce", "lapce", "main", []),
    ("Kong", "insomnia", "main", []),
    ("ant-design", "ant-design-pro", "main", []),
    ("naptha", "tesseract.js", "main", []),
    ("dotnet", "aspnetcore", "main", []),
    ("denysdovhan", "wtfjs", "main", []),
    ("httpie", "cli", "main", []),
    ("FreeCodeCampChina", "freecodecamp.cn", "main", []),
    ("huihut", "interview", "main", []),
    ("TeamNewPipe", "NewPipe", "main", []),
    ("kilimchoi", "engineering-blogs", "main", []),
    ("eugenp", "tutorials", "main", []),
    ("exacity", "deeplearningbook-chinese", "main", []),
    ("alibaba", "arthas", "main", []),
    ("huiyadanli", "RevokeMsgPatcher", "main", []),
    ("openai", "gym", "main", []),
    ("exelban", "stats", "main", []),
    ("ShareX", "ShareX", "main", []),
    ("YunaiV", "ruoyi-vue-pro", "main", []),
    ("doocs", "leetcode", "main", []),
    ("amruthpillai", "reactive-resume", "main", []),
    ("mattermost", "mattermost", "main", []),
    ("veggiemonk", "awesome-docker", "main", []),
    ("jaywcjlove", "linux-command", "main", []),
    ("jondot", "awesome-react-native", "main", []),
    ("airbnb", "lottie-android", "main", []),
    ("junegunn", "vim-plug", "main", []),
    ("vadimdemedes", "ink", "main", []),
    ("xyflow", "xyflow", "main", []),
    ("ueberdosis", "tiptap", "main", []),
    ("unknwon", "the-way-to-go_ZH_CN", "main", []),
    ("pbatard", "rufus", "main", []),
    ("bumptech", "glide", "main", []),
    ("geekan", "HowToLiveLonger", "main", []),
    ("zsh-users", "zsh-autosuggestions", "main", []),
    ("sorrycc", "awesome-javascript", "main", []),
    ("typicode", "husky", "main", []),
    ("netty", "netty", "main", []),
    ("geekcomputers", "Python", "main", []),
    ("qishibo", "AnotherRedisDesktopManager", "main", []),
    ("sxyazi", "yazi", "main", []),
    ("zxing", "zxing", "main", []),
    ("GorvGoyl", "Clone-Wars", "main", []),
    ("halfrost", "LeetCode-Go", "main", []),
    ("CMU-Perceptual-Computing-Lab", "openpose", "main", []),
    ("alibaba", "easyexcel", "main", []),
    ("Blankj", "AndroidUtilCode", "main", []),
    ("labstack", "echo", "main", []),
    ("facebookresearch", "fairseq", "main", []),
    ("frappe", "erpnext", "main", []),
    ("ashishpatel26", "500-AI-Machine-learning-Deep-learning-Computer-vision-NLP-Projects-with-code", "main", []),
    ("casey", "just", "main", []),
    ("nicolargo", "glances", "main", []),
    ("lovell", "sharp", "main", []),
    ("linexjlin", "GPTs", "main", []),
    ("hasura", "graphql-engine", "main", []),
    ("helix-editor", "helix", "main", []),
    ("astral-sh", "ruff", "main", []),
]