/*======================================================================
  POSTGRESQL 15+ · ESQUEMA COMPLETO "Cualificación de Pernos" · Versión 8
  ----------------------------------------------------------------------
  Novedades v8 (agosto 2025)
    • baseline.is_valid BOOLEAN DEFAULT TRUE  ← nueva columna para marcar
      cada captura como válida o inválida.
======================================================================*/

BEGIN;

/*--------------------------------------------------------------------
  0. Limpieza previa (idempotente)
--------------------------------------------------------------------*/
DROP VIEW  IF EXISTS batch_view_ordered               CASCADE;
DROP TABLE IF EXISTS baseline                         CASCADE;
DROP TABLE IF EXISTS bending_measurement              CASCADE;
DROP TABLE IF EXISTS bending_loading                  CASCADE;
DROP TABLE IF EXISTS one10_measurement                CASCADE;
DROP TABLE IF EXISTS one10_loading                    CASCADE;
DROP TABLE IF EXISTS temp_valid_combo                 CASCADE;
DROP TABLE IF EXISTS temp_measurement                 CASCADE;
DROP TABLE IF EXISTS temp_loading                     CASCADE;
DROP TABLE IF EXISTS temp_tof_loading                 CASCADE;
DROP TABLE IF EXISTS one4_valid_combo                 CASCADE;
DROP TABLE IF EXISTS one4_measurement                 CASCADE;
DROP TABLE IF EXISTS one4_loading                     CASCADE;
DROP TABLE IF EXISTS pre_valid_combo                  CASCADE;
DROP TABLE IF EXISTS pre_measurement                  CASCADE;
DROP TABLE IF EXISTS bolt_alias                       CASCADE;
DROP TABLE IF EXISTS bolt                             CASCADE;
DROP TABLE IF EXISTS customer_batch                   CASCADE;
DROP TABLE IF EXISTS batch                            CASCADE;
DROP TABLE IF EXISTS customer                         CASCADE;

/*--------------------------------------------------------------------
  1. Clientes y lotes
--------------------------------------------------------------------*/
CREATE TABLE customer (
    customer_id  SERIAL PRIMARY KEY,
    name         VARCHAR(60) NOT NULL UNIQUE
);

CREATE TABLE batch (
    batch_id                 VARCHAR(40)  PRIMARY KEY,
    customer                 VARCHAR(40),
    metric                   VARCHAR(20),
    length                   DOUBLE PRECISION,
    ultrasonic_length        DOUBLE PRECISION,
    grade                    VARCHAR(20),
    manufacturer             VARCHAR(40),
    customer_part_number     VARCHAR(40),
    additional_comment       TEXT,
    application_description  TEXT,
    nut_or_tapped_hole       VARCHAR(15),
    joint_length             DOUBLE PRECISION,
    max_load                 DOUBLE PRECISION,
    target_load              DOUBLE PRECISION,
    min_load                 DOUBLE PRECISION,
    min_temp                 DOUBLE PRECISION,
    max_temp                 DOUBLE PRECISION,
    frequency                DOUBLE PRECISION,
    gain                     DOUBLE PRECISION,
    cycles_coarse            INTEGER,
    cycles_fine              INTEGER,
    temperature              DOUBLE PRECISION,
    reference_tof            DOUBLE PRECISION,
    temp_gradient            DOUBLE PRECISION DEFAULT -103,
    short_temporal_window    INTEGER,
    short_signal_power_first_window INTEGER DEFAULT 196,
    long_temporal_window     INTEGER,
    long_signal_power_first_window  INTEGER DEFAULT 506,
    short_correlation_window INTEGER,
    long_correlation_window  INTEGER DEFAULT 10,
    temporal_signal_power    INTEGER DEFAULT 0,
    correlation_signal_power INTEGER DEFAULT 0,
    xi                       DOUBLE PRECISION,
    alpha1                   DOUBLE PRECISION,
    alpha2                   DOUBLE PRECISION,
    alpha3                   DOUBLE PRECISION,
    created_at               TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE customer_batch (
    customer_id  INTEGER     REFERENCES customer(customer_id) ON DELETE CASCADE,
    batch_id     VARCHAR(40) REFERENCES batch(batch_id)       ON DELETE CASCADE,
    PRIMARY KEY (customer_id, batch_id)
);

/*--------------------------------------------------------------------
  2. Pernos e alias corto
--------------------------------------------------------------------*/
CREATE TABLE bolt (
    batch_id   VARCHAR(40) REFERENCES batch(batch_id) ON DELETE CASCADE,
    bolt_id    VARCHAR(40) NOT NULL,
    PRIMARY KEY (batch_id, bolt_id)
);

CREATE TABLE bolt_alias (
    batch_id   VARCHAR(40),
    bolt_id    VARCHAR(40),
    bolt_num   SMALLINT NOT NULL CHECK (bolt_num BETWEEN 1 AND 255),
    PRIMARY KEY (batch_id, bolt_num),
    UNIQUE (batch_id, bolt_id),
    FOREIGN KEY (batch_id, bolt_id)
        REFERENCES bolt(batch_id, bolt_id)
        ON DELETE CASCADE
);

/*--------------------------------------------------------------------
  3. Baseline  (ahora con is_valid)
--------------------------------------------------------------------*/
CREATE TABLE baseline (
    id           BIGSERIAL PRIMARY KEY,
    batch_id     VARCHAR(40),
    bolt_id      VARCHAR(40),
    pallet_num   SMALLINT NOT NULL CHECK (pallet_num BETWEEN 1 AND 999),
    is_valid     BOOLEAN DEFAULT FALSE,
    freq         INTEGER,
    gain         INTEGER,
    pulse        INTEGER,
    pico1        DOUBLE PRECISION,
    pct_diff     DOUBLE PRECISION,
    tof          DOUBLE PRECISION,
    temp         DOUBLE PRECISION,
    force        DOUBLE PRECISION,
    maxcorrx     DOUBLE PRECISION,
    maxcorry     DOUBLE PRECISION,
    dat2         BYTEA,
    dat3         BYTEA,
    measured_at  TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (batch_id, bolt_id)
        REFERENCES bolt(batch_id, bolt_id) ON DELETE CASCADE
);
CREATE INDEX ix_baseline_bolt    ON baseline (batch_id, bolt_id);
CREATE INDEX ix_baseline_pallet  ON baseline (pallet_num);

/*--------------------------------------------------------------------
  4. Fase PRE
--------------------------------------------------------------------*/
CREATE TABLE pre_measurement (
    id           BIGSERIAL PRIMARY KEY,
    batch_id     VARCHAR(40),
    bolt_id      VARCHAR(40),
    freq         INTEGER,
    gain         INTEGER,
    pulse        INTEGER,
    pico1        DOUBLE PRECISION,
    pct_diff     DOUBLE PRECISION,
    tof          DOUBLE PRECISION,
    temp         DOUBLE PRECISION,
    force        DOUBLE PRECISION,
    maxcorrx     DOUBLE PRECISION,
    maxcorry     DOUBLE PRECISION,
    dat2         BYTEA,
    dat3         BYTEA,
    measured_at  TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (batch_id, bolt_id)
        REFERENCES bolt(batch_id, bolt_id) ON DELETE CASCADE
);

CREATE TABLE pre_valid_combo (
    id           BIGSERIAL PRIMARY KEY,
    batch_id     VARCHAR(40) REFERENCES batch(batch_id) ON DELETE CASCADE,
    freq         INTEGER,
    gain         INTEGER,
    pulse        INTEGER,
    is_best      BOOLEAN DEFAULT FALSE,
    created_at   TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (batch_id, freq, gain, pulse)
);

/*--------------------------------------------------------------------
  5. Fase 1-4
--------------------------------------------------------------------*/
CREATE TABLE one4_loading (
    id           BIGSERIAL PRIMARY KEY,
    batch_id     VARCHAR(40),
    bolt_id      VARCHAR(40),
    seq          INTEGER,
    tof          DOUBLE PRECISION,
    force        DOUBLE PRECISION,
    dat2         BYTEA,
    dat3         BYTEA,
    measured_at  TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (batch_id, bolt_id)
        REFERENCES bolt(batch_id, bolt_id) ON DELETE CASCADE
);

CREATE TABLE one4_measurement (
    id           BIGSERIAL PRIMARY KEY,
    batch_id     VARCHAR(40),
    bolt_id      VARCHAR(40),
    freq         INTEGER,
    gain         INTEGER,
    pulse        INTEGER,
    pico1        DOUBLE PRECISION,
    pct_diff     DOUBLE PRECISION,
    tof          DOUBLE PRECISION,
    temp         DOUBLE PRECISION,
    force        DOUBLE PRECISION,
    maxcorrx     DOUBLE PRECISION,
    maxcorry     DOUBLE PRECISION,
    dat2         BYTEA,
    dat3         BYTEA,
    measured_at  TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (batch_id, bolt_id)
        REFERENCES bolt(batch_id, bolt_id) ON DELETE CASCADE
);

CREATE TABLE one4_valid_combo (
    id           BIGSERIAL PRIMARY KEY,
    batch_id     VARCHAR(40) REFERENCES batch(batch_id) ON DELETE CASCADE,
    freq         INTEGER,
    gain         INTEGER,
    pulse        INTEGER,
    is_best      BOOLEAN DEFAULT FALSE,
    created_at   TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (batch_id, freq, gain, pulse)
);

/*--------------------------------------------------------------------
  6. Fase TEMP
--------------------------------------------------------------------*/
-- Serie temporal de estabilización de ToF durante la fase de temperatura
-- (sin dat2/dat3). Permite reproducir el gráfico de carga.
CREATE TABLE temp_tof_loading (
    id           BIGSERIAL PRIMARY KEY,
    batch_id     VARCHAR(40),
    bolt_id      VARCHAR(40),
    seq          INTEGER,                 -- índice del escalón de temperatura
    setpoint     DOUBLE PRECISION,        -- consigna de temperatura (°C)
    oven_temp    DOUBLE PRECISION,        -- temperatura medida (°C)
    tof          DOUBLE PRECISION,        -- ToF medido (ns)
    freq         INTEGER,
    gain         INTEGER,
    pulse        INTEGER,
    measured_at  TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (batch_id, bolt_id)
        REFERENCES bolt(batch_id, bolt_id) ON DELETE CASCADE
);

CREATE TABLE temp_loading (
    id           BIGSERIAL PRIMARY KEY,
    batch_id     VARCHAR(40),
    bolt_id      VARCHAR(40),
    seq          INTEGER,
    tof          DOUBLE PRECISION,
    temp         DOUBLE PRECISION,
    dat2         BYTEA,
    dat3         BYTEA,
    measured_at  TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (batch_id, bolt_id)
        REFERENCES bolt(batch_id, bolt_id) ON DELETE CASCADE
);

CREATE TABLE temp_measurement (
    id           BIGSERIAL PRIMARY KEY,
    batch_id     VARCHAR(40),
    bolt_id      VARCHAR(40),
    freq         INTEGER,
    gain         INTEGER,
    pulse        INTEGER,
    pico1        DOUBLE PRECISION,
    pct_diff     DOUBLE PRECISION,
    tof          DOUBLE PRECISION,
    temp         DOUBLE PRECISION,
    maxcorrx     DOUBLE PRECISION,
    maxcorry     DOUBLE PRECISION,
    dat2         BYTEA,
    dat3         BYTEA,
    measured_at  TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (batch_id, bolt_id)
        REFERENCES bolt(batch_id, bolt_id) ON DELETE CASCADE
);

CREATE TABLE temp_valid_combo (
    id           BIGSERIAL PRIMARY KEY,
    batch_id     VARCHAR(40) REFERENCES batch(batch_id) ON DELETE CASCADE,
    freq         INTEGER,
    gain         INTEGER,
    pulse        INTEGER,
    is_best      BOOLEAN DEFAULT FALSE,
    created_at   TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (batch_id, freq, gain, pulse)
);

/*--------------------------------------------------------------------
  7. Fase 1‑10
--------------------------------------------------------------------*/
CREATE TABLE one10_loading (
    id           BIGSERIAL PRIMARY KEY,
    batch_id     VARCHAR(40),
    bolt_id      VARCHAR(40),
    seq          INTEGER,
    tof          DOUBLE PRECISION,
    force        DOUBLE PRECISION,
    dat2         BYTEA,
    dat3         BYTEA,
    measured_at  TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (batch_id, bolt_id)
        REFERENCES bolt(batch_id, bolt_id) ON DELETE CASCADE
);

CREATE TABLE one10_measurement (
    id                BIGSERIAL PRIMARY KEY,
    batch_id          VARCHAR(40),
    bolt_id           VARCHAR(40),
    stage             TEXT CHECK (stage IN ('initial','final')),
    freq              INTEGER,
    gain              INTEGER,
    pulse             INTEGER,
    pico1             DOUBLE PRECISION,
    pct_diff          DOUBLE PRECISION,
    tof               DOUBLE PRECISION,
    temp              DOUBLE PRECISION,
    force_load_cell   DOUBLE PRECISION,
    maxcorrx          DOUBLE PRECISION,
    maxcorry          DOUBLE PRECISION,
    dat2              BYTEA,
    dat3              BYTEA,
    measured_at       TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (batch_id, bolt_id)
        REFERENCES bolt(batch_id, bolt_id) ON DELETE CASCADE
);
CREATE INDEX ix_one10_meas_bolt ON one10_measurement (batch_id, bolt_id);

/*--------------------------------------------------------------------
  8. Fase BENDING
--------------------------------------------------------------------*/
CREATE TABLE bending_loading (
    id           BIGSERIAL PRIMARY KEY,
    batch_id     VARCHAR(40) NOT NULL,
    bolt_id      VARCHAR(40) NOT NULL,
    position     SMALLINT NOT NULL CHECK (position BETWEEN 1 AND 3),
    seq          INTEGER,
    tof          DOUBLE PRECISION,
    force        DOUBLE PRECISION,
    dat2         BYTEA,
    dat3         BYTEA,
    measured_at  TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (batch_id, bolt_id)
        REFERENCES bolt(batch_id, bolt_id) ON DELETE CASCADE
);
CREATE INDEX ix_bending_load_bolt_pos ON bending_loading (batch_id, bolt_id, position);

CREATE TABLE bending_measurement (
    id                BIGSERIAL PRIMARY KEY,
    batch_id          VARCHAR(40) NOT NULL,
    bolt_id           VARCHAR(40) NOT NULL,
    position          SMALLINT NOT NULL CHECK (position BETWEEN 1 AND 3),
    tof               DOUBLE PRECISION,
    force             DOUBLE PRECISION,
    force_load_cell   DOUBLE PRECISION,
    stage             TEXT CHECK (stage IN ('initial','final')),
    freq              INTEGER,
    gain              INTEGER,
    pulse             INTEGER,
    pico1             DOUBLE PRECISION,
    pct_diff          DOUBLE PRECISION,
    temp              DOUBLE PRECISION,
    maxcorrx          DOUBLE PRECISION,
    maxcorry          DOUBLE PRECISION,
    dat2              BYTEA,
    dat3              BYTEA,
    measured_at       TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (batch_id, bolt_id)
        REFERENCES bolt(batch_id, bolt_id) ON DELETE CASCADE,
    UNIQUE (batch_id, bolt_id, position, measured_at)
);
CREATE INDEX ix_bending_bolt      ON bending_measurement (batch_id, bolt_id);
CREATE INDEX ix_bending_position  ON bending_measurement (position);


/*--------------------------------------------------------------------
  9. Vista ordenada (sin cambios, solo lee batch)
--------------------------------------------------------------------*/
CREATE VIEW batch_view_ordered AS
SELECT
  batch_id,
  customer,
  metric,
  length,
  ultrasonic_length,
  grade,
  manufacturer,
  customer_part_number,
  additional_comment,
  application_description,
  nut_or_tapped_hole,
  joint_length,
  max_load,
  target_load,
  min_load,
  min_temp,
  max_temp,
  frequency,
  gain,
  cycles_coarse,
  cycles_fine,
  temperature,
  reference_tof,
  temp_gradient,
  short_temporal_window,
  short_signal_power_first_window,
  long_temporal_window,
  long_signal_power_first_window,
  short_correlation_window,
  long_correlation_window,
  temporal_signal_power,
  correlation_signal_power,
  xi,
  alpha1,
  alpha2,
  alpha3,
  created_at
FROM batch;

COMMIT;
/* Fin del esquema v8 */
