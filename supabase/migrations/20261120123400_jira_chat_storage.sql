CREATE TABLE IF NOT EXISTS public.conversations (
    user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    id text NOT NULL,
    session_id text,
    title text NOT NULL DEFAULT '',
    markdown text NOT NULL DEFAULT '',
    messages_snapshot jsonb NOT NULL DEFAULT '[]'::jsonb,
    context_snapshot jsonb,
    drafts_snapshot jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    expires_at timestamptz NOT NULL,
    PRIMARY KEY (user_id, id)
);

CREATE INDEX IF NOT EXISTS conversations_user_updated_idx
    ON public.conversations (user_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS conversations_expiry_idx
    ON public.conversations (expires_at);

ALTER TABLE public.conversations ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can read their own conversations"
    ON public.conversations FOR SELECT
    USING (auth.uid() = user_id);
CREATE POLICY "Users can create their own conversations"
    ON public.conversations FOR INSERT
    WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Users can update their own conversations"
    ON public.conversations FOR UPDATE
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Users can delete their own conversations"
    ON public.conversations FOR DELETE
    USING (auth.uid() = user_id);

CREATE TABLE IF NOT EXISTS public.session_files (
    session_id text NOT NULL,
    file_id text NOT NULL,
    storage_path text NOT NULL,
    expires_at timestamptz NOT NULL,
    PRIMARY KEY (session_id, file_id)
);

CREATE INDEX IF NOT EXISTS session_files_expiry_idx
    ON public.session_files (expires_at);

ALTER TABLE public.session_files ENABLE ROW LEVEL SECURITY;

INSERT INTO storage.buckets (
    id,
    name,
    public,
    file_size_limit,
    allowed_mime_types
)
VALUES (
    'user-context-files',
    'user-context-files',
    false,
    10485760,
    ARRAY['text/plain']
)
ON CONFLICT (id) DO UPDATE
SET public = false,
    file_size_limit = EXCLUDED.file_size_limit,
    allowed_mime_types = EXCLUDED.allowed_mime_types;
