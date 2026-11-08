/** Match the subscription targets a request may reach, including Arena choices. */
export const modelMayUseSubscriptions = (
	model: Record<string, any> | undefined,
	models: Array<Record<string, any>>,
	seen = new Set<string>()
): boolean => {
	if (!model) return false;
	const id = model.id ?? '';
	if (seen.has(id)) return false;
	const visited = new Set([...seen, id]);
	const base = model.base_model_id ?? model.info?.base_model_id ?? model.info?.meta?.base_model_id;
	if (
		id.startsWith('subscription:') || model.owned_by === 'subscription' ||
		model.connection_type === 'subscription' || model.subscription ||
		base?.startsWith('subscription:')
	) return true;
	if (base && modelMayUseSubscriptions(models.find((m) => m.id === base), models, visited)) return true;
	if (model.owned_by !== 'arena') return false;
	const meta = model.info?.meta ?? {};
	const ids: string[] = Array.isArray(meta.model_ids) ? meta.model_ids : [];
	const candidates = models.filter((m) => m.owned_by !== 'arena' && (
		ids.length === 0 || (meta.filter_mode === 'exclude' ? !ids.includes(m.id) : ids.includes(m.id))
	));
	return candidates.some((m) => modelMayUseSubscriptions(m, models, visited));
};

export const requestMayUseSubscriptions = (
	modelIds: string[],
	models: Array<Record<string, any>>
): boolean => modelIds.some((id) =>
	id.startsWith('subscription:') || modelMayUseSubscriptions(models.find((m) => m.id === id), models)
);
