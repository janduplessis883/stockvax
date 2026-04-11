create table if not exists public.app_config (
  key text primary key,
  value text not null,
  created_at timestamptz not null default timezone('utc', now()),
  updated_at timestamptz not null default timezone('utc', now())
);

create or replace function public.set_app_config_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = timezone('utc', now());
  return new;
end;
$$;

drop trigger if exists trg_app_config_updated_at on public.app_config;
create trigger trg_app_config_updated_at
before update on public.app_config
for each row
execute function public.set_app_config_updated_at();

insert into public.app_config (key, value)
values ('qr_base_url', 'https://stockvax-stanhope.streamlit.app')
on conflict (key) do nothing;
